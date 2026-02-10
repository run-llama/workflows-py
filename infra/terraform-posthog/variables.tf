# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

variable "posthog_api_key" {
  description = "PostHog personal API key"
  type        = string
  sensitive   = true
}

variable "posthog_project_id" {
  description = "PostHog project ID (environment) to target"
  type        = string
}

variable "slack_bot_token" {
  description = "Slack Bot OAuth token (xoxb-...) with chat:write scope"
  type        = string
  sensitive   = true
}

variable "slack_channel" {
  description = "Slack channel ID to post feedback notifications to"
  type        = string
}
