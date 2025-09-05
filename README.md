Replica of Mixture of Cognitive Reasoners (MiCRo);

This project is a clean-room replication of the *Mixture of Cognitive Reasoners (MiCRo)* paper by Sabbata et al. (2025), which proposed a novel architecture combining routing and modular reasoning inspired by human cognition.

Project Highlights:

- Modular Cognitive Architecture: Implements MiCRo’s design with 4 expert modules (Language, Logic, Social, World)
- Custom Router: Trained a routing module to direct tokens to the most relevant expert
- Full Finetuning Pipeline: Integrated MiCRo into a T5-style encoder and finetuned on instruction-following datasets (e.g., Dolly, FLAN)
- Routing Heatmaps: Visualized which tokens were routed to which experts for interpretability
- Expert Ablation: Code structured to disable individual experts to study impact

Sample:
Prompt: "Lily said she was fine after the test, but her tone suggested otherwise."
Routing:

"fine"           → Social

"test"           → Logic

"tone"           → Social

"suggested"      → Lang

"otherwise"      → World

Link to the paper: https://arxiv.org/abs/2506.13331


