# > Internship in OneFlow

This is the summary of my internship experience at OneFlow.

TL;NR: I helped with compiling Stable Diffusion models by OneFlow.

## > What is OneFlow? Uhhh, I mean, as a framework but not a company?

OneFlow, a company known for its flagship product, a high-speed AI compiler, shares the same name as its main offering. In essence, OneFlow the framework seamlessly integrates with PyTorch, providing comprehensive support for compiling PyTorch models and enabling further optimizations.

In the open-source GitHub repository for onediff (a drop-in acceleration library for ComfyUI, HF diffusers, Stable Diffusion web UI, and other diffusion models), there are clear examples on how to use OneFlow for compilation.

And here is a piece of an instance:

```Python
base = StableDiffusionXLPipeline.from_pretrained(
    args.base,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")
base.unet = oneflow_compile(base.unet, options={"debug": 0})
imageprocessor = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
)
image = imageprocessor.images
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")
```

`oneflow_compile` wraps the a `torch.module` and concealedly creates a OneFlow graph to substitute for the original `torch.module`. In calling `base` pipeline (this call is conventially named **inference**), `base.unet` finds itself as a OneFlow graph and falls into OneFlow workflow.

In the OneFlow workflow, the primary task is compilation, which involves optimizing the graph by fusing, replacing, and removing nodes. This process also includes rewriting the low-level computational logic with the assistance of third-party libraries like MLIR (Multi-Level Intermediate Representation).

## > Internship begins, but with bugs.

My first job is about resolving a shape-related bug.

### > Bug description

Let me continue writing the above example code by adding one more inference

```python
imageprocessor = base(
    prompt=[args.prompt, args.prompt],
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
)
image = imageprocessor.images
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")
```

at the end.

Different from the above code, the new code snippet change the input from one prompt to two prompts, thus extending the batch size.
Then the bug shows up with such error message: `(4, 1280) != (2, 1280), shape mismatch`. Interestingly, this bug will not be triggered if I remove the first inference, implying that the culprit stays inbetween the two inferences.

After each inference, a compiled graph will be stored for potential further use to reduce unnecessary costs. A graph can be interpreted as a function receiving a matrix A[axb] and generate another matrix B[axc].

To understand this bug, I construct the following partial graph. Red color represents the data flow where the incorrect shape is introduced. Op `model.time_embedding.linear_2-fused_matmul_bias-20` meets such mismatch.

![](pic/oneflow1.png)
