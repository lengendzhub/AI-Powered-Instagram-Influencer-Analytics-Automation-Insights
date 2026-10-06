# Sensitivity Analysis — Ranking Stability

## Base Weights
```
{
  "engagement": 0.4,
  "engagement_residual": 0.2,
  "log_followers": 0.15,
  "posting_consistency": 0.15,
  "automation_probability": 0.1
}
```

## Base Top 10
| Rank | Handle |
|------|--------|
| 1 | @itsluxeindia_809 |
| 2 | @roamjournal35_2099 |
| 3 | @themodequeen_606 |
| 4 | @codetales68_93 |
| 5 | @power_by_priya_1404 |
| 6 | @thestrongqueen_1014 |
| 7 | @explore2_2024 |
| 8 | @theactiveking_1254 |
| 9 | @thezenking_1974 |
| 10 | @grace_of_deepak_1801 |

## Perturbation Results

| Perturbation | Overlap with Base Top 10 | Stability |
|-------------|--------------------------|----------|
| 1 | 10/10 | ✅ Stable |
| 2 | 10/10 | ✅ Stable |
| 3 | 10/10 | ✅ Stable |
| 4 | 10/10 | ✅ Stable |

> **Interpretation:** If 7+ of 10 handles remain in the top 10 across
> perturbations, the ranking is considered stable.
