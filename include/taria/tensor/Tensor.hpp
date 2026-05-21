#pragma once

#include <cstdint>
#include <vector>
#include <memory>

namespace taria {
namespace tensor {

/// Enum defining primitive tensor data types supported by Taria.
enum class DataType : uint8_t {
    FP64,
    FP32,
    FP16,
    BF16,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT8
};

/// Represents the physical memory layout of a Tensor (e.g., Row-major, Col-major, Tiled).
enum class Layout : uint8_t {
    DenseRowMajor,
    DenseColMajor,
    TiledZCurve
};

/// Tensor shape metadata, designed for fast access and MLIR shape inference compatibility.
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> d) : dims(d) {}

    int64_t num_elements() const {
        int64_t count = 1;
        for (auto d : dims) count *= d;
        return count;
    }
};

/// A hardware-agnostic Tensor abstraction representing both Host and Device memory.
///
/// Ownership Model:
/// Tensors manage their own underlying buffer lifetimes using a reference-counted
/// Pinned/Unified memory allocator provided by the `MemoryPool`.
///
/// Performance Implications:
/// Metadata (Shape, DataType) is stored on the host, but the `data_ptr` points
/// directly to hardware-aligned buffers to ensure 128-byte warp transaction coalescing.
class Tensor {
public:
    Tensor(Shape shape, DataType dtype, Layout layout = Layout::DenseRowMajor);
    ~Tensor();

    // Disable copy to enforce strict ownership transitions; allow move.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    void* data() const { return data_ptr_; }
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    size_t size_bytes() const;

    /// Moves tensor data synchronously to the active GPU context.
    void to_device();
    /// Moves tensor data synchronously to Host CPU memory.
    void to_host();

private:
    Shape shape_;
    DataType dtype_;
    Layout layout_;
    void* data_ptr_ = nullptr;
    bool is_device_ = false;
};

} // namespace tensor
} // namespace taria
