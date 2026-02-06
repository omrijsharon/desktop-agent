#!/usr/bin/env node
import OpenAI from "openai";

function usage() {
  console.error(
    "Usage: OPENAI_API_KEY=... node scripts/openai_web_search_demo.mjs \"Your question\" [--model gpt-5] [--show-sources]",
  );
}

function parseArgs(argv) {
  const args = { query: "", model: "gpt-5", showSources: false };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--model") {
      args.model = argv[i + 1] || args.model;
      i++;
      continue;
    }
    if (a === "--show-sources") {
      args.showSources = true;
      continue;
    }
    rest.push(a);
  }
  args.query = rest.join(" ").trim();
  return args;
}

const args = parseArgs(process.argv.slice(2));
if (!args.query) {
  usage();
  process.exit(2);
}

if (!process.env.OPENAI_API_KEY) {
  console.error("ERROR: OPENAI_API_KEY is not set.");
  process.exit(2);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const response = await client.responses.create({
  model: args.model,
  tools: [{ type: "web_search" }],
  input: args.query,
});

console.log(response.output_text || "");

if (args.showSources) {
  const sources = [];
  for (const item of response.output || []) {
    if (item?.type === "web_search_call") {
      for (const s of item?.action?.sources || []) {
        if (s?.url) sources.push(s.url);
      }
    }
  }
  const deduped = [...new Set(sources)];
  if (deduped.length) {
    console.log("\nSources:");
    for (const u of deduped) console.log(`- ${u}`);
  }
}

