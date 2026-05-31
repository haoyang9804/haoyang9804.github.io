function wrapPreNode(preNode) {
  return {
    type: "element",
    tagName: "details",
    properties: {
      className: ["code-details"],
      open: true
    },
    children: [
      {
        type: "element",
        tagName: "summary",
        properties: {
          className: ["code-summary"]
        },
        children: [
          {
            type: "element",
            tagName: "span",
            properties: {
              className: ["code-summary-label"]
            },
            children: [{ type: "text", value: "Code" }]
          }
        ]
      },
      preNode
    ]
  };
}

function visitChildren(node) {
  if (!Array.isArray(node.children)) {
    return;
  }

  node.children = node.children.map((child) => {
    if (child.type === "element" && child.tagName === "pre") {
      return wrapPreNode(child);
    }

    visitChildren(child);
    return child;
  });
}

export default function rehypeCodeDetails() {
  return (tree) => {
    visitChildren(tree);
  };
}
