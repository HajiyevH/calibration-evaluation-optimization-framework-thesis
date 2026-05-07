#pragma once
#include <string>
#include "contracts/types.h"
#include "contracts/status.h"

namespace ba {

// Parse problem.json into BundleAdjustmentProblem + BundleAdjustmentParams
Status loadProblemJson(const std::string& path,
                       BundleAdjustmentProblem& problem,
                       BundleAdjustmentParams& params);

// Write result.json from BundleAdjustmentResult, original problem, and solver params
Status writeResultJson(const std::string& path,
                       const BundleAdjustmentResult& result,
                       const BundleAdjustmentProblem& original_problem,
                       const BundleAdjustmentParams& params = {});

} // namespace ba
