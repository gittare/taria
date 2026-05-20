#ifndef TARIA_DIALECT_H
#define TARIA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the dialect definition.
#include "TariaDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "TariaEnums.h.inc"

#define GET_OP_CLASSES
#include "TariaOps.h.inc"

#endif // TARIA_DIALECT_H
