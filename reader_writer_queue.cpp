#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

static size_t chunk_size_s = 100;
static size_t num_chunks_s = 1000;
static size_t queue_depth_s = 500;
static size_t num_preloaders_s = 2;
static size_t num_readers_s = 5;

/// Simple sequential sampler.
class Sampler {
public:
  Sampler() : current_index_(0) {}

  std::vector<size_t> get_index_batch(size_t batch_size) {
    size_t size = std::min((size_ - current_index_), batch_size);
    std::vector<size_t> res(size);
    std::iota(std::begin(res), std::end(res), current_index_);
    current_index_ += size;
    return res;
  }

  /// This is what we really need.
  void reset(size_t size) {
    current_index_ = 0;
    size_ = size;
  }

private:
  size_t size_;
  size_t current_index_;
};

/// Struct to hold chunk information.
struct ChunkData {
  ChunkData(size_t chk_idx, size_t chk_size, Sampler s, std::vector<int> data)
      : chunk_index(chk_idx),
        remaining_example_count(chk_size),
        sampler(std::move(s)) {
    chunk_data = std::move(data);
  }
  size_t chunk_index;
  size_t remaining_example_count;
  Sampler sampler;
  std::vector<int> chunk_data;
};

/// Main class that handles the chunk data.
class ChunkDataBuffer {
public:
  ChunkDataBuffer(size_t num_chunks)
      : remaining_chunk_count_(num_chunks), total_example_count_in_queue_(0) {}

  /// Multi-reader multi writer buffer.
  std::vector<int> get_batch(size_t batch_size) {
    std::vector<int> res;
    size_t count = 0;

    while (count < batch_size) {
      std::unique_lock<std::mutex> lock(mutex_);
      cvr_.wait(lock, [this] { // readers wait till these two conditions.
        return (
            this->total_example_count_in_queue_ > 0 ||
            remaining_chunk_count_ == 0);
      });
      if (remaining_chunk_count_ == 0) {
        lock.unlock();
        // cvw_.notify_all();
        return res; // unless a reset is done, data read is already completed.
      }
      while (count < batch_size && chunk_queue_.size() > 0) {
        size_t local_count = 0;
        auto& chk_data = chunk_queue_.front();
        if (chk_data.remaining_example_count > 0) {
          for (size_t i :
               chk_data.sampler.get_index_batch(batch_size - count)) {
            res.emplace_back(chk_data.chunk_data[i]);
            count++;
            local_count++;
          }
          chk_data.remaining_example_count -= local_count;
          total_example_count_in_queue_ -= local_count;
        }
        assert(chk_data.remaining_example_count >= 0);
        if (chk_data.remaining_example_count == 0) {
          chunk_queue_.pop();
          remaining_chunk_count_--;
        }
      }
      lock.unlock();
      cvw_.notify_all(); // notify all writers.
    }
    return res;
  }

  /// Preload threads call this method to add data.
  void add_chunk_data(size_t index, std::vector<int> data) {
    std::unique_lock<std::mutex> lock(mutex_);
    cvw_.wait(lock, [this] { // writers wait for this condition.
      return this->total_example_count_in_queue_ < queue_depth_s;
    });
    Sampler sampler; // in the real dataset, we need to get a copy of the
                     // sampler and reset it.
    sampler.reset(data.size());
    ChunkData chk_data(index, data.size(), sampler, data);
    chunk_queue_.push(std::move(chk_data));
    total_example_count_in_queue_ += data.size();
    lock.unlock();
    cvr_.notify_all(); // notify all readers.
  }

  size_t total_example_count_in_queue_;
  size_t remaining_chunk_count_;
  std::queue<ChunkData> chunk_queue_;
  std::mutex mutex_;
  std::condition_variable cvr_;
  std::condition_variable cvw_;
};

class ChunkDataSet {
public:
  ChunkDataSet(size_t num_chunks)
      : num_chunks_(num_chunks), chunks_to_load_(num_chunks) {}

  ~ChunkDataSet() {
    for (auto& t : preload_threads_) {
      t.join();
    }
  }

  void preloader(size_t id) {
    while (true) {
      size_t chunk_id = -1;
      {
        std::lock_guard<std::mutex> lock(
            mutex_); // This is simply the mutex for generating chunk index. We
                     // can wrap the chunk sampler using the thread-safe sampler
                     // to achieve the same effect.
        if (chunks_to_load_ > 0) {
          chunk_id = --chunks_to_load_;
        } else {
          break;
        }
      }

      if (chunk_id >= 0) {
        // std::cout << "PRELOADER " << id << " ADDING CHUNK ID " << chunk_id
        //          << std::endl;
        chunk_buffer_->add_chunk_data(chunk_id, read_chunk(chunk_id));
      }
    }
    std::cout << "preloader stopping :" << id << std::endl;
  }

  /// user facing API.
  std::vector<int> read_chunk(int index) {
    return std::vector<int>(chunk_size_s);
  }

  /// This is what we override and it should be simple like this.
  std::vector<int> get_batch(size_t batch_size) {
    if (chunk_buffer_ == nullptr) {
      throw std::runtime_error(
          "Dataset has not been reset() before calling get_batch().");
    }
    return chunk_buffer_->get_batch(batch_size);
  }

  /// This is our init method. I included this in the PR to FB.
  void reset() {
    chunks_to_load_ = num_chunks_;
    chunk_buffer_ = std::make_unique<ChunkDataBuffer>(
        num_chunks_); // Creates a new chunk buffer each time we reset the
                      // dataset.
    for (size_t i = 0; i < num_preloaders_s; ++i) {
      preload_threads_.emplace_back(
          [this, i]() mutable { this->preloader(i); });
    }
  }

private:
  std::unique_ptr<ChunkDataBuffer> chunk_buffer_;
  std::vector<std::thread> preload_threads_;
  std::mutex mutex_;
  size_t num_chunks_;
  size_t chunks_to_load_;
};

class DataLoader {
public:
  DataLoader(std::shared_ptr<ChunkDataSet> dataset, int num_threads)
      : dataset_(dataset), num_threads_(num_threads), total_examples_read(0) {}

  void read_data(int id) {
    while (!stop_) {
      std::vector<int> batch = dataset_->get_batch(32);
      total_examples_read.fetch_add(batch.size(), std::memory_order_relaxed);
      // std::cout << "Loader thread: " << id << " read " << batch.size()
      //          << " examples." << std::endl;
      if (batch.size() == 0) {
        std::cout << "read_data stopping :" << id << std::endl;
        break;
      }
    }
  }

  void start() {
    std::cout << "Dataloader starting..." << std::endl;
    dataset_->reset();
    stop_ = false;
    for (int i = 0; i < num_threads_; ++i) {
      threads_.emplace_back([this, i]() mutable { this->read_data(i); });
    }
  }

  void wait() {
    for (auto& t : threads_) {
      t.join();
    }
    assert(total_examples_read == num_chunks_s * chunk_size_s);
    std::cout << "Dataloader stopping..." << std::endl;
    std::cout << "Total examples read = " << total_examples_read
              << " It should match with = " << num_chunks_s * chunk_size_s
              << std::endl;
  }

private:
  std::shared_ptr<ChunkDataSet> dataset_;
  int num_threads_;
  std::vector<std::thread> threads_;
  bool stop_;
  std::mutex tmp_mutex_;

  // Just for validating.
  std::atomic<size_t> total_examples_read;
};

int main() {
  std::shared_ptr<ChunkDataSet> dsp =
      std::make_shared<ChunkDataSet>(num_chunks_s);
  DataLoader loader(dsp, num_readers_s);
  loader.start();
  loader.wait();
  int y;
  std::cin >> y;
  return 0;
}
