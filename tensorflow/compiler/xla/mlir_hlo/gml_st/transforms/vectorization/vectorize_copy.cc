/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/vectorization/vectorization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_VECTORIZECOPYPASS
#include "gml_st/transforms/passes.h.inc"

/// Custom vectorization pattern for small and non-contiguous memref::CopyOp.
struct CopyVectorizationPattern : public OpRewritePattern<memref::CopyOp> {
  CopyVectorizationPattern(MLIRContext *context, int64_t numElementsThreshold,
                           mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<memref::CopyOp>(context, benefit),
        numElementsThreshold(numElementsThreshold) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(op.getTarget().getType());

    if (!srcType || !targetType) return failure();

    if (!srcType.hasStaticShape() || !targetType.hasStaticShape())
      return failure();

    // If memref has an identity layout or is contiguous with an arbitrary
    // offset, it will be turned into llvm.memcpy intrinsic later, do not
    // vectorize it.
    if (memref::isStaticShapeAndContiguousRowMajor(srcType) &&
        memref::isStaticShapeAndContiguousRowMajor(targetType)) {
      return failure();
    }

    auto isSmallMemrefType = [&](MemRefType memrefType) {
      return memrefType.getNumElements() > 0 &&
             memrefType.getNumElements() < numElementsThreshold;
    };

    // If memref is too big, vectorizing it actually explodes the compilation
    // time. Also, ignore empty memrefs, which will be handled by memrefCopy
    // function.
    if (!isSmallMemrefType(srcType) || !isSmallMemrefType(targetType)) {
      return failure();
    }
    return linalg::vectorizeCopy(rewriter, op);
  }

 private:
  int64_t numElementsThreshold;
};

struct VectorizeCopyPass
    : public impl::VectorizeCopyPassBase<VectorizeCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<CopyVectorizationPattern>(ctx, numElementsThreshold);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass(
    int64_t numElementsThreshold) {
  VectorizeCopyPassOptions opts;
  opts.numElementsThreshold = numElementsThreshold;
  return std::make_unique<VectorizeCopyPass>(opts);
}

}  // namespace gml_st
}  // namespace mlir
