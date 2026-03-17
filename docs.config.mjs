export default {
  section: "llama-index-workflows",
  label: "LlamaAgents",
  content: [
    { src: "./docs/src/content/docs/llamaagents", dest: "python/llamaagents" },
  ],
  sidebar: [
    {
      label: "LlamaAgents",
      content: {
        type: "autogenerate",
        directory: "python/llamaagents",
        collapsed: true,
      },
      append: [
        {
          label: "Agent Workflows Reference \u{1F517}",
          link: "/python/workflows-api-reference/",
        },
      ],
    },
  ],
};
