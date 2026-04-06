Phase 2 of the experimental plan focuses on **improving the practical utility and interpretability** of inDecay to satisfy the requirements of Reviewers 2 and 3. This phase transitions the model from a statistical success to a transparent tool with immediate biological application.

### **1. Granular Frameshift Outcome Breakdown**
To move beyond the general $R^2$ for frameshift ratios, you will perform a detailed analysis of specific mutational outcomes.
*   **Experimental Task:** Using the existing identifiers and indel type used by the model, you will aggregate the predicted and observed frequencies into **three distinct categories: +1, +2, and +3 (or in-frame) outcomes**.
*   **Technical Detail:** This involves mapping the high-resolution event probabilities predicted by the $f_p$ module specifically to their resulting reading frames.
*   **Utility:** You will calculate the **prediction accuracy (e.g., $R^2$ or KL divergence)** for these specific frame ratios across all tested cell lines and zygote datasets. This allows researchers to optimize for the most efficient functional knockout by choosing target sites that favor specific frameshift types.

### **2. Sophisticated Interpretability through SHAP Analysis**
Reviewer 2 noted that the current ablation study, which masks feature dimensions to observe performance drops, is too basic. You will implement **SHAP (Shapley Additive Explanations)** to provide a more granular view of feature influence.
*   **Feature Focus:** The analysis will specifically target the **61 information-dense features**. This includes:
    *   **13 deletion-specific features:** Such as the raw deletion length ($l_{ld}$), deletion start site ($ss$), and the various decay terms derived from first-principle knowledge.
    *   **7 insertion-specific features:** Including insertion length and the presence of complementary nucleotides near the cut site.
    *   **41 shared features:** Such as the GC-ratio of the gRNA, total microhomology (MH) strength, and the one-hot encoded 9-bp sequence upstream of the PAM.
*   **Goal:** By applying SHAP to the trained multilayer perceptron (MLP), you will generate **summary plots** showing exactly how features like "maximum MH length" or "proximity to cut site" drive the probability of specific major repair events.

### **3. Locus-Level Embryo Viability and Quality Control**
To address concerns regarding potential selection bias in the zygote mutational profiles, you must provide more transparent quality control metrics.
*   **Data Aggregation:** You will create a comprehensive table for the **52 mouse target sites** and **12 livestock loci**.
*   **Specific Metrics:** For each target, the report must include:
    *   **Attrition Rates:** The number of initial zygotes (from the total pool of 11,787 for mouse or 3,632 for livestock) vs. the number that survived to the blastocyst stage or overnight culture.
    *   **Success Rates:** The proportion of surviving embryos that were "successfully edited" (e.g., the 1,878 successful mouse embryos).
    *   **Sequencing Quality:** A report on the $r^2$ signal quality from Sanger sequencing for each locus to justify the selection of high-quality test gRNAs.
*   **Rationale:** This detailed reporting will prove that the observed mutational preferences—such as the bias toward 1-bp edits in mouse zygotes—are a result of biological repair pathways rather than an artifact of **differential survival at specific loci**.

By completing these steps, the manuscript will provide the "sophisticated and granular understanding" requested by the reviewers, significantly enhancing the transparency of the neural network's predictions.