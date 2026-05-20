#include "TariaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

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

// Dialect interfaces, canonicalization, etc. would go here
