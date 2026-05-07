#pragma once
#include "contracts/types.h"
#include "contracts/status.h"

namespace ba {

class BundleAdjuster {
public:
    Status solve(const BundleAdjustmentProblem& problem,
                 const BundleAdjustmentParams& params,
                 BundleAdjustmentResult& result);
};

} // namespace ba
