# Blogging

Add posts as Markdown files under `src/content/blog`.

You can duplicate `src/content/blog/_template.md`, rename it, and set `draft: false` when it is ready to publish.

```md
---
title: "Post title"
description: "Optional one-sentence summary."
pubDate: 2026-05-22
updatedDate: 2026-05-22
tags: ["llm-infra", "cuda"]
draft: false
---

Write the post body here.
```

The filename becomes the URL. For example:

- `src/content/blog/softmax-cuda.md` -> `/blog/softmax-cuda/`
- `src/content/blog/infra/rl.md` -> `/blog/infra/rl/`

Set `draft: true` to keep a post out of the generated blog and RSS feed.
