# Docs QA evaluation

Label: **dev, optimistic**

Default configuration: `v1`

## Metrics by split and configuration

```json
{
  "category_counts": {
    "dev": {
      "v1": {
        "fact": 10,
        "multi_chunk": 1,
        "out_of_scope": 6,
        "rare_keyword": 1,
        "reasoning": 6
      },
      "v2a": {
        "fact": 10,
        "multi_chunk": 1,
        "out_of_scope": 6,
        "rare_keyword": 1,
        "reasoning": 6
      },
      "v2b": {
        "fact": 10,
        "multi_chunk": 1,
        "out_of_scope": 6,
        "rare_keyword": 1,
        "reasoning": 6
      },
      "v2b_norerank": {
        "fact": 10,
        "multi_chunk": 1,
        "out_of_scope": 6,
        "rare_keyword": 1,
        "reasoning": 6
      }
    },
    "held_out": {
      "v1": {
        "fact": 0,
        "multi_chunk": 0,
        "out_of_scope": 0,
        "rare_keyword": 0,
        "reasoning": 0
      },
      "v2a": {
        "fact": 0,
        "multi_chunk": 0,
        "out_of_scope": 0,
        "rare_keyword": 0,
        "reasoning": 0
      },
      "v2b": {
        "fact": 0,
        "multi_chunk": 0,
        "out_of_scope": 0,
        "rare_keyword": 0,
        "reasoning": 0
      },
      "v2b_norerank": {
        "fact": 0,
        "multi_chunk": 0,
        "out_of_scope": 0,
        "rare_keyword": 0,
        "reasoning": 0
      }
    }
  },
  "category_scores": {
    "dev": {
      "v1": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2a": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2b": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2b_norerank": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      }
    },
    "held_out": {
      "v1": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2a": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2b": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      },
      "v2b_norerank": {
        "fact": {},
        "multi_chunk": {},
        "out_of_scope": {},
        "rare_keyword": {},
        "reasoning": {}
      }
    }
  },
  "comparisons_by_split": {
    "dev": {},
    "held_out": {}
  },
  "false_refusal_rate": {
    "dev": {
      "v1": 0.05555555555555555,
      "v2a": 0.0,
      "v2b": 0.05555555555555555,
      "v2b_norerank": 0.16666666666666666
    },
    "held_out": {
      "v1": null,
      "v2a": null,
      "v2b": null,
      "v2b_norerank": null
    }
  },
  "judge_agreement": {},
  "judge_correlation": null,
  "metrics_by_split": {
    "dev": {
      "v1": {},
      "v2a": {},
      "v2b": {},
      "v2b_norerank": {}
    },
    "held_out": {
      "v1": {},
      "v2a": {},
      "v2b": {},
      "v2b_norerank": {}
    }
  },
  "nan_counts": {},
  "refusal_accuracy": {
    "dev": {
      "v1": 1.0,
      "v2a": 1.0,
      "v2b": 1.0,
      "v2b_norerank": 1.0
    },
    "held_out": {
      "v1": null,
      "v2a": null,
      "v2b": null,
      "v2b_norerank": null
    }
  },
  "repeat_variance": {},
  "retrieval_by_split": {
    "dev": {
      "v1": {
        "precision_at_k": 0.32222222222222224,
        "recall_at_k": 0.8888888888888888,
        "reciprocal_rank": 0.9555555555555556
      },
      "v2a": {
        "precision_at_k": 0.34444444444444444,
        "recall_at_k": 0.9166666666666666,
        "reciprocal_rank": 0.9444444444444444
      },
      "v2b": {
        "precision_at_k": 0.35555555555555557,
        "recall_at_k": 0.8611111111111112,
        "reciprocal_rank": 0.8888888888888888
      },
      "v2b_norerank": {
        "precision_at_k": 0.2777777777777778,
        "recall_at_k": 0.8611111111111112,
        "reciprocal_rank": 0.8472222222222222
      }
    },
    "held_out": {
      "v1": {
        "precision_at_k": null,
        "recall_at_k": null,
        "reciprocal_rank": null
      },
      "v2a": {
        "precision_at_k": null,
        "recall_at_k": null,
        "reciprocal_rank": null
      },
      "v2b": {
        "precision_at_k": null,
        "recall_at_k": null,
        "reciprocal_rank": null
      },
      "v2b_norerank": {
        "precision_at_k": null,
        "recall_at_k": null,
        "reciprocal_rank": null
      }
    }
  },
  "retrieval_comparisons_by_split": {
    "dev": {
      "v2a-v1": {
        "precision_at_k": {
          "ci": [
            0.0,
            0.05555555555555556
          ],
          "mean_diff": 0.022222222222222223,
          "verdict": "no_detectable_difference"
        },
        "recall_at_k": {
          "ci": [
            0.0,
            0.08333333333333333
          ],
          "mean_diff": 0.027777777777777776,
          "verdict": "no_detectable_difference"
        },
        "reciprocal_rank": {
          "ci": [
            -0.12222222222222222,
            0.10555555555555557
          ],
          "mean_diff": -0.011111111111111108,
          "verdict": "no_detectable_difference"
        }
      },
      "v2b-v1": {
        "precision_at_k": {
          "ci": [
            -0.011111111111111112,
            0.07777777777777778
          ],
          "mean_diff": 0.03333333333333333,
          "verdict": "no_detectable_difference"
        },
        "recall_at_k": {
          "ci": [
            -0.08333333333333333,
            0.0
          ],
          "mean_diff": -0.027777777777777776,
          "verdict": "no_detectable_difference"
        },
        "reciprocal_rank": {
          "ci": [
            -0.15,
            0.0
          ],
          "mean_diff": -0.06666666666666667,
          "verdict": "no_detectable_difference"
        }
      },
      "v2b-v2a": {
        "precision_at_k": {
          "ci": [
            -0.022222222222222223,
            0.05555555555555556
          ],
          "mean_diff": 0.011111111111111112,
          "verdict": "no_detectable_difference"
        },
        "recall_at_k": {
          "ci": [
            -0.16666666666666666,
            0.0
          ],
          "mean_diff": -0.05555555555555555,
          "verdict": "no_detectable_difference"
        },
        "reciprocal_rank": {
          "ci": [
            -0.16666666666666666,
            0.0
          ],
          "mean_diff": -0.05555555555555555,
          "verdict": "no_detectable_difference"
        }
      },
      "v2b_norerank-v1": {
        "precision_at_k": {
          "ci": [
            -0.08888888888888889,
            -0.011111111111111112
          ],
          "mean_diff": -0.044444444444444446,
          "verdict": "worse"
        },
        "recall_at_k": {
          "ci": [
            -0.08333333333333333,
            0.0
          ],
          "mean_diff": -0.027777777777777776,
          "verdict": "no_detectable_difference"
        },
        "reciprocal_rank": {
          "ci": [
            -0.21944444444444444,
            -0.022222222222222223
          ],
          "mean_diff": -0.10833333333333334,
          "verdict": "worse"
        }
      }
    },
    "held_out": {
      "v2a-v1": {
        "precision_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "recall_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "reciprocal_rank": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        }
      },
      "v2b-v1": {
        "precision_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "recall_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "reciprocal_rank": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        }
      },
      "v2b-v2a": {
        "precision_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "recall_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "reciprocal_rank": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        }
      },
      "v2b_norerank-v1": {
        "precision_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "recall_at_k": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        },
        "reciprocal_rank": {
          "ci": [
            null,
            null
          ],
          "mean_diff": null,
          "verdict": "inconclusive"
        }
      }
    }
  },
  "thresholds_by_config": {
    "v1": {
      "context_recall": null,
      "faithfulness": null
    },
    "v2a": {
      "context_recall": null,
      "faithfulness": null
    },
    "v2b": {
      "context_recall": null,
      "faithfulness": null
    },
    "v2b_norerank": {
      "context_recall": null,
      "faithfulness": null
    }
  }
}
```
