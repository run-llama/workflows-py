# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

provider "posthog" {
  api_key    = var.posthog_api_key
  project_id = var.posthog_project_id
}

resource "posthog_hog_function" "slack_feedback_notification" {
  name        = "Slack Feedback Notification"
  description = "Posts to Slack whenever a feedback form is submitted"
  type        = "destination"
  enabled     = true
  icon_url    = "https://raw.githubusercontent.com/PostHog/posthog/master/frontend/public/services/slack.png"

  # Custom Hog code — based on PostHog's template-slack but uses a direct bot
  # token instead of an OAuth integration, so it can be fully driven by vars.
  hog = <<-EOT
    let res := fetch('https://slack.com/api/chat.postMessage', {
      'method': 'POST',
      'headers': {
        'Authorization': f'Bearer {inputs.slack_bot_token}',
        'Content-Type': 'application/json'
      },
      'body': {
        'channel': inputs.slack_channel,
        'icon_emoji': ':mega:',
        'username': 'PostHog Feedback',
        'blocks': inputs.blocks,
        'text': inputs.text
      }
    })

    if (res.status >= 400) {
      throw Error(f'Error from Slack API: {res.status}: {res.body}');
    }
    if (res.body.ok != true) {
      throw Error(f'Error from Slack API: {res.body}');
    }
  EOT

  inputs_json = jsonencode({
    slack_bot_token = { value = var.slack_bot_token }
    slack_channel   = { value = var.slack_channel }
    text = {
      value = "New feedback submitted by *{person.name}*"
    }
    blocks = {
      value = [
        {
          type = "section"
          text = {
            type = "mrkdwn"
            text = ":mega: *New Feedback Submitted*"
          }
        },
        {
          type = "section"
          fields = [
            { type = "mrkdwn", text = "*From:*\n{person.name}" },
            { type = "mrkdwn", text = "*Event:*\n{event.event}" }
          ]
        },
        {
          type = "actions"
          elements = [
            {
              type = "button"
              text = {
                type = "plain_text"
                text = "View in PostHog"
              }
              url = "{person.url}"
            }
          ]
        }
      ]
    }
  })

  filters_json = jsonencode({
    events = [
      {
        id   = "survey sent"
        name = "survey sent"
        type = "events"
      }
    ]
  })
}
