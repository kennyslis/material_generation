# Interview Delivery Notes

## 1. What this project demonstrates

This repository is positioned as an engineering prototype for inverse design of 2D HER catalysts. The project emphasizes three things that interviewers usually care about:

- a complete end-to-end pipeline instead of a partial notebook,
- a model architecture that is technically aligned with the task statement,
- reproducible outputs including weights, figures, candidate structures, and comparison metrics.

## 2. Key message to say during the interview

A concise way to present the work:

> I implemented a conditional diffusion pipeline over 2D crystal graphs. The denoiser uses a lightweight GNN to model composition and atomic coordinates jointly, and generation is followed by a multi-objective reranking stage that explicitly optimizes HER activity, stability, and synthesizability proxies. To keep the repository runnable in a constrained environment, I packaged a lightweight surrogate dataset generator while leaving clear interfaces for connecting real databases like C2DB, JARVIS-DFT, and 2DMatPedia.

## 3. What is genuinely strong in this repo

- The code is runnable as-is and not just conceptual.
- The repo contains train/test entry points, saved checkpoints, plots, and generated structures.
- The architecture matches the interview prompt: diffusion, GNN backbone, HER optimization, stability optimization, synthesizability prediction, and multi-task loss.
- The README includes diagrams, formulas, comparison metrics, and reproduction commands.

## 4. What to proactively clarify if asked

- The current labels are proxy/surrogate labels rather than DFT-computed ground truth.
- The current generated structures are 2D coordinate candidates, not final DFT-relaxed crystal structures.
- The baseline comparison inside this repo is a controlled baseline-style setting, not an official reproduction of the external GitHub repository.

That is still acceptable for an interview project, as long as it is stated clearly and confidently.

## 5. Recommended wording when explaining the data choice

> Because the interview task required a complete runnable repository, I first built a surrogate-data version to validate the whole algorithmic pipeline end to end. The dataset module is intentionally structured so the surrogate generator can be replaced by real crystal records from C2DB or JARVIS without changing the training or inference stack.

## 6. Best next upgrade if more time is available

- Replace the surrogate dataset builder with real 2D material records.
- Use pymatgen or ASE for crystal parsing and export CIF/POSCAR files.
- Add a property predictor pretrained on public 2D materials data.
- Add novelty and diversity metrics across generated candidates.
- Report relaxation or stability verification with external simulation tools.
