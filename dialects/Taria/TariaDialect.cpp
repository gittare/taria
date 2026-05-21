#include "TariaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::taria;

//===----------------------------------------------------------------------===//
// Taria dialect
//===----------------------------------------------------------------------===//

void TariaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TariaOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CompressOp implementation
//===----------------------------------------------------------------------===//

LogicalResult CompressOp::inferReturnTypes(MLIRContext *context,
                                           std::optional<Location> location,
                                           ValueRange operands,
                                           DictionaryAttr attributes,
                                           OpaqueProperties properties,
                                           RegionRange regions,
                                           SmallVectorImpl<Type> &inferredReturnTypes) {
    // For Taria semantic compression, we expect the output type of a compress block
    // to be an int8 quantized bitstream tensor.
    // In a real pass we might compute the dimension based on entropy ratios.
    // Here we stub out the logic:
    if (operands.empty()) {
        return failure();
    }

    Type inputTy = operands[0].getType();
    if (auto tensorTy = dyn_cast<RankedTensorType>(inputTy)) {
        // Output is a 1D bitstream for demonstration
        inferredReturnTypes.push_back(RankedTensorType::get({ShapedType::kDynamic}, IntegerType::get(context, 8)));
        return success();
    }

    return failure();
}

// Dialect interfaces and canonicalization patterns are implemented here...
