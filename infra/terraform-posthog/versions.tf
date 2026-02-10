# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

terraform {
  required_version = ">= 1.0"

  required_providers {
    posthog = {
      source  = "PostHog/posthog"
      version = "~> 1.0.4"
    }
  }
}
