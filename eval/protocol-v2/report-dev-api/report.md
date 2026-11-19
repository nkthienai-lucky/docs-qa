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
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.775973521863352,
            "context_precision": 0.9999999999,
            "context_recall": 0.9,
            "faithfulness": 0.8666666666666667
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.8405533833325328,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.9458777899222226,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.7087531707138568,
            "context_precision": 0.83333333325,
            "context_recall": 0.8333333333333334,
            "faithfulness": 0.8333333333333334
          }
        }
      },
      "v2a": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.7345073688515373,
            "context_precision": 0.8888888887999999,
            "context_recall": 0.9,
            "faithfulness": 0.8866666666666667
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.8405533833325328,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 1.0
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.9458777899222226,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.837779655601797,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        }
      },
      "v2b": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.7507588236715442,
            "context_precision": 0.89999999991,
            "context_recall": 0.9,
            "faithfulness": 0.98
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": 0.0,
            "faithfulness": 1.0
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.9458777899222226,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.6959681136228139,
            "context_precision": 0.83333333325,
            "context_recall": 0.8333333333333334,
            "faithfulness": 1.0
          }
        }
      },
      "v2b_norerank": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.6802143187774919,
            "context_precision": 0.8888888887999999,
            "context_recall": 0.8,
            "faithfulness": 0.8166666666666667
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.8405533833325328,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.9458777899222226,
            "context_precision": 0.9999999999,
            "context_recall": 1.0,
            "faithfulness": 1.0
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": 0.43461413110012226,
            "context_precision": 0.6666666666,
            "context_recall": 0.6666666666666666,
            "faithfulness": 0.6666666666666666
          }
        }
      }
    },
    "held_out": {
      "v1": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        }
      },
      "v2a": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        }
      },
      "v2b": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        }
      },
      "v2b_norerank": {
        "fact": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "multi_chunk": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "out_of_scope": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "rare_keyword": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        },
        "reasoning": {
          "api_claude_sonnet_5_5": {
            "answer_relevancy": null,
            "context_precision": null,
            "context_recall": null,
            "faithfulness": null
          }
        }
      }
    }
  },
  "comparisons_by_split": {
    "dev": {
      "v2a-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.023137067989601606,
                0.17024702538367623
              ],
              "mean_diff": 0.04621447377939912,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.249999999975,
                0.1249999999875
              ],
              "mean_diff": -0.06249999999375,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.16666666666666666,
                0.16666666666666666
              ],
              "mean_diff": 0.0,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.014814814814814824,
                0.1962962962962963
              ],
              "mean_diff": 0.06666666666666667,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.0510057462275339,
                0.0019763166552506585
              ],
              "mean_diff": -0.019344548497902046,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.18749999998125,
                0.0
              ],
              "mean_diff": -0.06249999999375,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.16666666666666666,
                0.0
              ],
              "mean_diff": -0.05555555555555555,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                0.011111111111111108,
                0.2518518518518519
              ],
              "mean_diff": 0.11851851851851852,
              "verdict": "better"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b-v2a": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.18455986375394345,
                0.014690447135107943
              ],
              "mean_diff": -0.05579266091440056,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.18749999998125,
                0.0
              ],
              "mean_diff": -0.06249999999375,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.16666666666666666,
                0.0
              ],
              "mean_diff": -0.05555555555555555,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                0.0,
                0.13333333333333336
              ],
              "mean_diff": 0.05185185185185186,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b_norerank-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.2965842933477734,
                -0.01214310511248639
              ],
              "mean_diff": -0.1445792371411671,
              "verdict": "worse"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.18749999998125,
                0.0
              ],
              "mean_diff": -0.06249999999375,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.2777777777777778,
                0.0
              ],
              "mean_diff": -0.1111111111111111,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                -0.22592592592592595,
                0.020370370370370365
              ],
              "mean_diff": -0.08333333333333334,
              "verdict": "no_detectable_difference"
            }
          },
          "verdict": "inconclusive"
        }
      }
    },
    "held_out": {
      "v2a-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b-v2a": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        }
      },
      "v2b_norerank-v1": {
        "answer_relevancy": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_precision": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "context_recall": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        },
        "faithfulness": {
          "judges": {
            "api_claude_sonnet_5_5": {
              "ci": [
                null,
                null
              ],
              "mean_diff": null,
              "verdict": "inconclusive"
            }
          },
          "verdict": "inconclusive"
        }
      }
    }
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
      "v1": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": 0.7665936342317453,
          "context_precision": 0.9411764704941177,
          "context_recall": 0.8888888888888888,
          "faithfulness": 0.8703703703703703
        }
      },
      "v2a": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": 0.7930730036048648,
          "context_precision": 0.8823529410882353,
          "context_recall": 0.8888888888888888,
          "faithfulness": 0.937037037037037
        }
      },
      "v2b": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": 0.7428985122573264,
          "context_precision": 0.8823529410882353,
          "context_recall": 0.8333333333333334,
          "faithfulness": 0.9888888888888889
        }
      },
      "v2b_norerank": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": 0.6220143970905783,
          "context_precision": 0.8235294116823529,
          "context_recall": 0.7777777777777778,
          "faithfulness": 0.787037037037037
        }
      }
    },
    "held_out": {
      "v1": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": null,
          "context_precision": null,
          "context_recall": null,
          "faithfulness": null
        }
      },
      "v2a": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": null,
          "context_precision": null,
          "context_recall": null,
          "faithfulness": null
        }
      },
      "v2b": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": null,
          "context_precision": null,
          "context_recall": null,
          "faithfulness": null
        }
      },
      "v2b_norerank": {
        "api_claude_sonnet_5_5": {
          "answer_relevancy": null,
          "context_precision": null,
          "context_recall": null,
          "faithfulness": null
        }
      }
    }
  },
  "nan_counts": {
    "api_claude_sonnet_5_5": {
      "answer_relevancy": 27,
      "context_precision": 28,
      "context_recall": 24,
      "faithfulness": 24
    }
  },
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
