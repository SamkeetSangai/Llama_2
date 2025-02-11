# LLaMA 2

This repository provides an implementation of the LLaMA model architecture, inspired by Meta's LLaMA 2.

The code focuses on the model's structure and inference capabilities, incorporating features such as:

- **KV Caching**: Enhances inference efficiency by storing and reusing key-value pairs from previous computations.
- **Grouped Query Attention**: Optimizes attention mechanisms by grouping queries, leading to improved performance.
- **Rotary Embeddings**: Applies rotational position embeddings to capture positional information effectively.

Please note that this repository does not include training code. Training large language models like LLaMA is computationally intensive and typically requires substantial resources beyond a local setup. Therefore, the model weights are directly loaded from Meta's pre-trained models.

## Model Architecture

The LLaMA model is built upon the transformer architecture, featuring multiple layers of self-attention and feed-forward networks. The incorporation of KV caching, grouped query attention, and rotary embeddings enhances its efficiency and performance during inference.

![LLaMA Model Architecture](https://github.com/user-attachments/assets/55ae63b9-6527-4296-92e1-3a4d60494eba)

*Figure: The LLaMA transformer architecture. Image credit: Umar Jamil.*

## Example Input and Output

## Example Input and Output

Below are examples demonstrating three prompt strategies: **Zero-Shot**, **One-Shot**, and **Few-Shot**.

### Zero-Shot

**Input Prompt:**
```plaintext
Explain the concept of quantum entanglement in simple terms.
```

**Model Output:**
```plaintext
Quantum entanglement is a phenomenon in which two or more particles become connected in such a way that their properties are correlated, regardless of the distance between them. This means that if something happens to one particle, it will instantly affect the other particle, regardless of how far apart they are.

Imagine you have two coins. If you flip one coin, the other coin will land on the same side, even if you flip it on the other side of the room. (Note: This analogy is simplified and does not capture all aspects of quantum entanglement.)
```

### One-Shot

**Input Prompt:**
```plaintext
If Apple was founded in Japan instead of the US, it would have likely focused on:
Example: Sony revolutionized consumer electronics with the Walkman.
Apple, in a Japanese context, would have likely...
```

**Model Output:**
```plaintext
Apple, founded in Japan, would have likely focused on creating innovative and user-friendly technology products that cater to the unique preferences and needs of the Japanese market. Here are some potential areas of focus:

1. **Mobile Payments:** Given Japan's traditionally cash-based society, Apple might have prioritized mobile payment solutions to simplify transactions.
2. **Smart Home Technology:** With a growing interest in integrated smart home devices, the company could have developed products like smart thermostats, lighting systems, and security solutions.
3. **Personalized Health and Wellness:** Recognizing the strong emphasis on health in Japan, Apple might have introduced advanced wearable devices for fitness tracking and health monitoring.
4. **Consumer Electronics Design:** Embracing Japan's renowned design aesthetics, Apple may have emphasized sleek, minimalist products that resonate with local tastes.
```

### Few-Shot

**Input Prompt:**
```plaintext
Translate English to French:
sunflower => tournesol
laptop => ordinateur portable
wireless headphones => écouteurs sans fil
jellyfish =>
tree => arbre
water => eau
mountain => montagne
river => fleuve
bird => oiseau
cat => chat
dog => chien
house => maison
car => voiture
park => parc
garden => jardin
city => ville
tree => arbre
flower => fleur
fish => poisson
beach => plage
sea => mer
sky => ciel
moon => lune
star => étoile
rain => pluie
cloud => nuage
wind => vent
snow => neige
ice => glace
fire => feu
water => eau
earth => terre
sun => soleil
moon => lune
sky => ciel
cloud => nuage
wind => vent
rain => pluie
snow => neige
```

**Model Output:**
```plaintext
sunflower => tournesol
laptop => ordinateur portable
wireless headphones => écouteurs sans fil
jellyfish => méduse
tree => arbre
water => eau
mountain => montagne
river => fleuve
bird => oiseau
cat => chat
dog => chien
house => maison
car => voiture
park => parc
garden => jardin
city => ville
tree => arbre
flower => fleur
fish => poisson
beach => plage
sea => mer
sky => ciel
moon => lune
star => étoile
rain => pluie
cloud => nuage
wind => vent
snow => neige
ice => glace
fire => feu
earth => terre
sun => soleil
```

These examples demonstrate the model's capability to perform language translation and provide informative responses based on the given prompts.

For more detailed information on the LLaMA 2 architecture and its features, please refer to Meta's official documentation.

## Credits

- **Meta**: For developing the original LLaMA 2 model and providing the pre-trained weights.
