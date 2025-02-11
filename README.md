# LLaMA 2

This repository provides an implementation of the LLaMA model architecture, inspired by Meta's LLaMA 2.

The code focuses on the model's structure and inference capabilities, incorporating features such as:

- **KV Caching**: Enhances inference efficiency by storing and reusing key-value pairs from previous computations.
- **Grouped Query Attention**: Optimizes attention mechanisms by grouping queries, leading to improved performance.
- **Rotary Embeddings**: Applies rotational position embeddings to capture positional information effectively.

Please note that this repository does not include training code. Training large language models like LLaMA is computationally intensive and typically requires substantial resources beyond a local setup. Therefore, the model weights are directly loaded from Meta's pre-trained models.

## Model Architecture

The LLaMA model is built upon the transformer architecture, featuring multiple layers of self-attention and feed-forward networks. The incorporation of KV caching, grouped query attention, and rotary embeddings enhances its efficiency and performance during inference.

![LLaMA Model Architecture](![image](https://github.com/user-attachments/assets/55ae63b9-6527-4296-92e1-3a4d60494eba)

*Figure: The LLaMA transformer architecture. Image credit: Umar Jamil.*

## Example Input and Output

**Input Prompt**:
```plaintext
Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe en peluche
cheese =>
```

**Model Output**:
```plaintext
fromage
```

These examples demonstrate the model's capability to perform language translation and provide informative responses based on the given prompts.

For more detailed information on the LLaMA 2 architecture and its features, please refer to Meta's official documentation.

## Credits

- **Meta**: For developing the original LLaMA 2 model and providing the pre-trained weights.v
