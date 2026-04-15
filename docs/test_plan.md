# Test Plan & Test Cases

## Acceptance Criteria
- All API endpoints return correct status codes
- Model predictions are positive numbers
- Demand classification returns valid status strings
- Batch predictions return correct count
- Driver recommendations return valid actions
- Drift detector correctly identifies COVID collapse

## Test Cases

| ID | Test | Expected | Result |
|---|---|---|---|
| TC01 | GET /health | 200, status=ok | PASS |
| TC02 | GET /ready | 200, status=ready | PASS |
| TC03 | POST /predict zone 161 rush hour | 200, demand > 0 | PASS |
| TC04 | POST /predict high lag values | demand_status in valid values | PASS |
| TC05 | POST /recommend 520 demand 25 drivers | recommended_drivers > 0 | PASS |
| TC06 | POST /predict/batch 2 zones | 2 predictions returned | PASS |
| TC07 | Drift score 2020-04 | score > 0.15 (got 0.77) | PASS |
| TC08 | Drift score 2019-01 | score < 0.15 (got 0.007) | PASS |

## Test Report
- Total test cases: 8
- Passed: 8
- Failed: 0
- Test framework: pytest
- Run command: python -m pytest tests/test_api.py -v

## How to Run Tests
cd /Users/rishwanthpb/MLOPS/MLOPS_PROJECT
python -m pytest tests/test_api.py -v