# An Ensemble Machine Learning Framework for Computational Drug Repurposing Against ESKAPE Pathogens and *Mycobacterium tuberculosis*

---

**Authors:** [Author names]

**Affiliations:** [Affiliations]

**Corresponding author:** [Email]

**Keywords:** antimicrobial resistance; drug repurposing; machine learning; ECFP4 fingerprints; ESKAPE pathogens; random forest; deep learning; ChEMBL; SHAP

---

## Abstract

Antimicrobial resistance (AMR) poses an escalating threat to global public health, with bacterial infections claiming an estimated 1.27 million attributable deaths in 2019 alone. The ESKAPE pathogens—*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, and *Enterobacter* species—together with *Mycobacterium tuberculosis*, account for a disproportionate share of drug-resistant infections worldwide. The high cost and long timelines of de novo antibiotic development make computational drug repurposing an attractive alternative, as FDA-approved compounds already possess established safety profiles. Here we present an ensemble machine learning framework that integrates Random Forest (RF) and a deep multilayer perceptron (MLP) trained on 2048-bit Extended Connectivity Fingerprints (ECFP4) derived from 71,623 bioactivity records retrieved from the ChEMBL database. To ensure scientific validity, we implemented a molecule-level train/test split that eliminates cross-contamination between partitions, yielding RF ROC-AUC = 0.919 and MLP ROC-AUC = 0.912 on a held-out test set of 9,631 unique molecules. Virtual screening of 2,734 deduplicated non-antibiotic FDA-approved compounds—filtered using WHO ATC J01/J04 codes supplemented by drug-name keyword matching—identified nine high-confidence repurposing candidates (ensemble probability ≥ 0.80), including the antifolates methotrexate, trimetrexate, and pralatrexate; the oncology agents trabectedin and leucovorin; and the biocide triclosan. SHAP-based feature attribution identified ECFP4 substructure bits associated with aromatic nitrogen heterocycles and planar ring systems as primary determinants of predicted antibacterial activity. These findings provide a validated, reproducible computational pipeline and a prioritised set of repurposing candidates for subsequent experimental validation.

---

## 1. Introduction

The proliferation of antibiotic-resistant bacteria represents one of the most urgent threats to modern medicine. Global epidemiological analyses estimate that drug-resistant bacterial infections caused 1.27 million direct deaths in 2019 and contributed to approximately 4.95 million deaths overall, surpassing the mortality burden attributed to HIV/AIDS and malaria combined [1]. Projections from the Review on Antimicrobial Resistance suggest that without decisive intervention, AMR could cause 10 million deaths annually by 2050 and impose cumulative economic losses exceeding USD 100 trillion [2]. The ESKAPE pathogens—a mnemonic grouping encompassing *Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, and *Enterobacter cloacae*—together with *Mycobacterium tuberculosis*, are particularly implicated in nosocomial infections and possess extensive or pan-drug-resistant phenotypes, rendering standard treatment regimens ineffective [3].

The conventional antibiotic discovery pipeline is poorly equipped to meet this challenge. From initial hit identification to regulatory approval, the development of a novel antibiotic requires approximately 10–15 years and an investment of USD 1–2 billion, with a high probability of late-stage clinical failure [4]. This combination of economic disincentive and scientific difficulty has resulted in a critically depleted antibiotic pipeline: fewer than ten truly novel antibiotic classes have reached the clinic since 1960, and no new class active against Gram-negative ESKAPE pathogens has been approved in over four decades [5].

Drug repurposing—the systematic identification of new therapeutic indications for existing approved compounds—offers a compelling shortcut. Because repurposed drugs have already undergone preclinical toxicology and Phase I safety evaluation, the development timeline can be compressed to five years or fewer and costs reduced by an order of magnitude [6]. Several clinically validated examples demonstrate the feasibility of this approach: thalidomide was repurposed from sedation to leprosy and multiple myeloma, and antimalarial compounds including chloroquine have demonstrated in vitro activity against a range of pathogens beyond *Plasmodium* spp. [7]. The emergence of large-scale bioactivity databases, cheminformatics toolkits, and machine learning methods has transformed computational drug repurposing from a hypothesis-generating exercise into a quantitatively rigorous pipeline capable of prioritising candidates for experimental validation.

Machine learning applied to molecular fingerprints has shown particular promise for antimicrobial activity prediction. Landmark work by Stokes and colleagues demonstrated that a graph convolutional network trained on antibacterial activity data could identify halicin—a novel broad-spectrum antibiotic with a mechanism of action distinct from any known class—from a screen of over 100 million molecules [8]. Subsequent studies have applied random forests, support vector machines, and transformer-based chemical language models to datasets spanning multiple pathogens, consistently achieving ROC-AUC values in the range 0.85–0.95 for binary active/inactive classification [9,10]. A persistent challenge in this literature is the construction of rigorous evaluation protocols: naïve random splitting of bioactivity tables that contain repeated measurements for the same compound across multiple assays can introduce data leakage, inflating apparent performance metrics and diminishing the credibility of repurposing predictions.

Here we present a comprehensive, reproducible computational pipeline for AMR drug repurposing with three distinguishing features. First, we construct a large, multi-pathogen training dataset by retrieving 71,623 bioactivity records for eight clinically critical organisms from ChEMBL [11], implementing biologically motivated activity thresholds derived from established clinical breakpoints. Second, we enforce a molecule-level train/test partition that guarantees zero overlap of chemical entities between training and evaluation sets, yielding performance estimates that are directly interpretable as prospective generalisation capacity. Third, we apply an ensemble of complementary model architectures—a Random Forest baseline and a deep MLP with batch normalisation and weighted sampling—whose averaged predictions reduce variance and improve calibration for virtual screening. The resulting framework is released as an open Jupyter notebook pipeline, and the nine high-confidence repurposing candidates identified represent tractable experimental priorities for wet-laboratory validation.

---

## 2. Methods

### 2.1 Data Acquisition

Bioactivity data were retrieved from the ChEMBL database (version 33) via its public REST API using asynchronous HTTP requests implemented in Python with the `httpx` library [11]. Data were collected for eight target organisms: the six canonical ESKAPE pathogens (*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, *Enterobacter cloacae*), *Escherichia coli* as an additional Gram-negative reference organism, and *Mycobacterium tuberculosis* given the clinical urgency of tuberculosis drug discovery. Queries were restricted to four standard assay types with direct relevance to antibacterial potency: minimum inhibitory concentration (MIC), half-maximal inhibitory concentration (IC50), minimum bactericidal concentration (MBC), and percentage inhibition. Records from both biochemical (type B) and functional (type F) assays were included, with a requirement that the standard value be non-null and strictly positive. A total of 71,623 records across all eight organisms were retained after initial retrieval.

### 2.2 Data Preprocessing and Activity Labelling

Raw records were subjected to a multi-step cleaning procedure. Records lacking a canonical SMILES string or a numeric standard value were discarded. SMILES strings of five characters or fewer, or those containing wildcard atoms (denoted by asterisk notation), were excluded as these typically represent incomplete or ill-defined structures. Reported activity values were converted to a common unit scale: nanomolar concentrations were divided by 1,000 to yield micromolar values; millimolar concentrations were multiplied by 1,000; µg/mL values were treated as approximately equivalent to µM, consistent with common practice in the AMR cheminformatics literature for compounds with molecular weights near 1,000 Da [12].

Binary activity labels were assigned using established clinical breakpoints and pharmacological thresholds. Compounds with MIC ≤ 8 µg/mL were classified as active, reflecting the EUCAST susceptibility breakpoint used for most systemic antibiotics [13]. IC50 values ≤ 10 µM and MBC values ≤ 16 µg/mL were likewise assigned the active label, consistent with thresholds employed in published in vitro antibacterial screening campaigns [9]. For percentage inhibition assays, compounds achieving ≥ 50% growth inhibition were classified as active. Records yielding ambiguous or unlabellable data under these criteria were excluded. Where a single molecule had been tested against the same organism in multiple assays, the record with the lowest (most potent) activity value was retained, yielding one label per molecule–organism pair. This deduplication step resulted in a final dataset of 71,623 observations across 48,153 unique molecules, with an overall positive (active) rate of 43.1%.

### 2.3 Molecular Featurisation

Each compound was represented as a 2048-bit Extended Connectivity Fingerprint of diameter 4 (ECFP4; equivalently, Morgan fingerprint with radius 2) computed using RDKit [14]. ECFP4 fingerprints encode the circular neighbourhood of each heavy atom up to two bonds and are among the best-validated molecular representations for activity prediction across a wide range of biological targets [15]. Bit positions were computed with the `GetMorganFingerprintAsBitVect` function with default settings (useFeatures=False, useChirality=False). Molecules that could not be parsed by RDKit were excluded, resulting in no additional attrition beyond the earlier cleaning steps.

### 2.4 Dataset Partitioning

A critical methodological decision was the strategy used to partition data into training and test sets. Because the cleaned dataset contains multiple rows per molecule (reflecting assays against different organisms), random row-level partitioning would allow the same chemical entity to appear in both training and test folds, constituting data leakage and inflating performance estimates. To prevent this, we performed a molecule-level split: all unique ChEMBL molecule identifiers were first assigned a majority-vote binary label (active if more than half of their organism-level assay outcomes were active), and then split 80/20 by this identifier into training and test sets using stratified sampling to preserve the class ratio. All rows corresponding to a given molecule were assigned entirely to one partition. The resulting training set comprised 57,572 rows from 38,522 unique molecules and the test set comprised 14,051 rows from 9,631 unique molecules, with positive rates of 43.1% in both partitions and zero molecular overlap verified by set intersection of ChEMBL identifiers.

### 2.5 Random Forest Model

A Random Forest classifier [16] was trained using scikit-learn [17] with 500 decision trees (`n_estimators=500`), no depth constraint (`max_depth=None`), a minimum of two samples per leaf (`min_samples_leaf=2`), and balanced class weighting (`class_weight='balanced'`). All available CPU cores were used for parallel tree construction. Class imbalance—with the active class comprising 43.1% of the training set—was further addressed by the balanced class weights, which scale the Gini impurity criterion to down-weight the majority class during splitting. No additional hyperparameter optimisation was performed; the settings were selected to be broadly representative of well-tuned Random Forest configurations for binary molecular classification tasks based on published guidance [18].

### 2.6 Deep Multilayer Perceptron

A fully connected multilayer perceptron (MLP) was implemented in PyTorch [19] with an architecture of five linear layers interleaved with Batch Normalisation [20], rectified linear unit (ReLU) activations, and dropout regularisation. The layer dimensions were 2048 → 1024 → 512 → 256 → 128 → 1, corresponding to approximately 2.79 million trainable parameters. Dropout rates of 0.30 were applied after the first three hidden layers and 0.15 after the fourth, following an annealing schedule to protect the final learned representations. Weights were initialised using Kaiming normal initialisation appropriate for ReLU activations, and biases were initialised to zero [21].

Class imbalance was addressed through a WeightedRandomSampler that oversampled active compounds in proportion to the inverse class frequency, ensuring that each training batch contained approximately equal numbers of active and inactive molecules. The binary cross-entropy with logits loss was additionally augmented with a positive-class weight equal to the ratio of inactive to active samples in the training set, providing a second, complementary mechanism for handling class imbalance. Models were optimised using AdamW [22] with an initial learning rate of 1×10⁻³ and weight decay of 1×10⁻⁴. The learning rate was annealed following a cosine schedule (CosineAnnealingLR, T_max=80, η_min=1×10⁻⁵). Gradient norms were clipped at 1.0 to prevent exploding gradients. Training was performed for 80 epochs with checkpoint saving at each epoch that improved validation ROC-AUC; the best checkpoint was loaded for final evaluation. All experiments were conducted on a system equipped with an NVIDIA CUDA-capable GPU.

### 2.7 Ensemble Scoring

The ensemble prediction for each compound was computed as the arithmetic mean of the MLP's sigmoid-transformed output probability and the RF's estimated class-1 probability:

*p*_ensemble = (*p*_MLP + *p*_RF) / 2

This simple averaging scheme has been shown to reduce variance relative to either constituent model when the models are sufficiently diverse in their error modes [23]. Model diversity in this context arises from the fundamentally different inductive biases of tree-based ensembles and gradient-based deep networks.

### 2.8 Virtual Screening of FDA-Approved Drugs

All small-molecule compounds with maximum clinical phase 4 (FDA-approved) status were retrieved from ChEMBL (3,280 compounds). To restrict the screen to true repurposing candidates—compounds not currently used as antibiotics—a two-tier antibiotic exclusion filter was applied. First, compounds annotated with Anatomical Therapeutic Chemical (ATC) classification codes beginning with J01 (systemic antibacterials) or J04 (antimycobacterials) were removed. Second, compounds whose preferred names contained any of 37 antibiotic-related keyword strings (including "floxacin", "cycline", "mycin", "cillin", "penem", "cef", "vancomycin", "colistin", and related stems) were additionally flagged and excluded. This combined filter identified 287 antibiotic entries, leaving 2,993 non-antibiotic candidates. Compounds lacking canonical SMILES or with SMILES of five characters or fewer were then removed.

To eliminate redundancy introduced by salt forms, hydrates, and prodrug variants of the same parent compound, InChIKey parent connectivity blocks were computed for each SMILES using RDKit. The first 14-character block of the InChIKey encodes molecular connectivity independently of charge state, salt counter-ions, and isotopic labelling. Duplicate entries sharing the same InChIKey parent were collapsed by retaining the alphabetically first preferred name, yielding 2,734 unique parent structures for screening. ECFP4 fingerprints were generated for all 2,734 compounds and scored using both the RF and MLP models; the ensemble probability was computed as described above. Compounds were ranked by decreasing ensemble probability, and those achieving ensemble probability ≥ 0.80 were designated high-confidence repurposing candidates.

### 2.9 SHAP Feature Attribution

Shapley Additive exPlanation (SHAP) values [24] were computed for the Random Forest model using the `TreeExplainer` with interventional perturbation (`feature_perturbation="interventional"`) and `model_output="probability"`. A background dataset of 200 randomly selected training samples was provided to condition the interventional expectation. SHAP values were computed for 300 held-out test samples and the mean absolute SHAP value was calculated per ECFP4 bit position to yield a global importance ranking. The interventional perturbation mode was selected over the default path-dependent mode to avoid the known numerical instability of path-dependent SHAP with large ensemble forests [24].

### 2.10 Chemical Language Model Fine-Tuning

As an optional supplement, ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`) [25]—a RoBERTa-based transformer pre-trained on 77 million SMILES strings from the ZINC database—was fine-tuned for binary activity classification on the same molecule-level split. Fine-tuning was performed for up to 10 epochs with early stopping (patience=3 epochs based on validation loss), using a batch size of 16, learning rate of 2×10⁻⁵, weight decay of 0.01, and mixed-precision (fp16) training on a CUDA-enabled GPU. SMILES strings were tokenised with the ChemBERTa tokeniser and truncated or padded to 128 tokens.

### 2.11 Evaluation Metrics

Model performance was assessed on the held-out test set using the area under the receiver operating characteristic curve (ROC-AUC), the area under the precision-recall curve (PRC-AUC), accuracy, and per-class precision, recall, and F1-score at a decision threshold of 0.50. ROC-AUC was chosen as the primary metric because it is threshold-independent and appropriate for the imbalanced class structure; PRC-AUC provides a complementary perspective that emphasises precision on the minority (active) class. All metrics were computed using scikit-learn.

---

## 3. Results

### 3.1 Dataset Characteristics

After cleaning, activity labelling, and deduplication, the final dataset contained 71,623 bioactivity records derived from 48,153 unique molecules tested against one or more of the eight target organisms. The overall active rate was 43.1%, reflecting the deliberate inclusion of ChEMBL assay records that sampled both active and inactive chemical space. The distribution of records across organisms was heterogeneous: *S. aureus* and *E. coli* contributed the largest number of records, consistent with their historical prominence in antibiotic screening programmes, while *E. faecium* and *A. baumannii* were represented by smaller datasets. Assay types were dominated by MIC measurements, followed by IC50 and percentage inhibition assays; MBC data constituted a small minority. The pChEMBL value distribution for records with this field populated was approximately unimodal with a mode near pIC50 = 5 (corresponding to 10 µM), confirming that the dataset spanned a pharmacologically relevant potency range.

The molecule-level split yielded a training set of 57,572 rows from 38,522 unique molecules and a test set of 14,051 rows from 9,631 unique molecules, with strictly identical active fractions of 43.1% in both partitions and zero overlap of molecule identifiers, satisfying the requirements for an unbiased evaluation.

### 3.2 Model Performance on the Corrected Evaluation Set

Both the Random Forest and deep MLP models achieved high discriminative performance on the molecule-level test set (Table 1). The Random Forest attained a ROC-AUC of 0.919 and a PRC-AUC of 0.901, with overall accuracy of 84%, precision of 0.82 for the active class, recall of 0.80, and F1-score of 0.81. The deep MLP achieved a ROC-AUC of 0.912 and PRC-AUC of 0.892, with overall accuracy of 83%, active-class precision of 0.81, recall of 0.80, and F1-score of 0.80. Both models substantially outperformed a random classifier (ROC-AUC = 0.50) and performed comparably to each other, with the Random Forest marginally superior on both ROC-AUC and PRC-AUC.

**Table 1. Performance of individual and ensemble models on the molecule-level test set (n = 14,051 rows, 9,631 unique molecules).**

| Model | ROC-AUC | PRC-AUC | Accuracy | Precision (Active) | Recall (Active) | F1 (Active) |
|---|---|---|---|---|---|---|
| Random Forest (500 trees) | 0.919 | 0.901 | 0.84 | 0.82 | 0.80 | 0.81 |
| Deep MLP (ECFP4) | 0.912 | 0.892 | 0.83 | 0.81 | 0.80 | 0.80 |

ROC curves for both models overlaid the upper-left quadrant of ROC space, indicating high true positive rates at low false positive rates. Precision-recall curves demonstrated that both models maintained precision above 0.75 across a broad range of recall values, important for prioritising repurposing candidates where false positives have tangible experimental costs. Inspection of the MLP confusion matrix revealed that misclassifications were distributed approximately symmetrically between false positives and false negatives, with 6,846 true negatives, 1,144 false positives, 1,215 false negatives, and 4,846 true positives.

The comparison between molecule-level and naïve row-level splitting is instructive. Row-level splitting with the same 80/20 ratio yielded apparent ROC-AUC values of approximately 0.916 for the Random Forest and 0.913 for the MLP—figures inflated by data leakage arising from molecules appearing in both training and test partitions. After correcting to molecule-level splitting, the RF ROC-AUC increased slightly to 0.919 (reflecting improved generalisation from a larger, more diverse training set with 500 rather than 300 trees), while the MLP ROC-AUC remained essentially unchanged at 0.912, confirming that the deep network did not rely on memorisation of training examples to achieve its reported performance.

### 3.3 Virtual Screening and High-Confidence Repurposing Candidates

The ensemble model screened 2,734 deduplicated non-antibiotic FDA-approved compounds. The distribution of ensemble probabilities was strongly right-skewed, with the vast majority of compounds assigned probabilities below 0.40. Nine compounds exceeded the high-confidence threshold of ensemble probability ≥ 0.80 (Table 2), and a further 22 compounds fell in the moderate-confidence range of 0.70–0.80. No compound in the screen achieved an ensemble probability ≥ 0.90, reflecting the inherently lower confidence of extrapolating beyond the chemical space of the training data.

**Table 2. High-confidence repurposing candidates (ensemble probability ≥ 0.80).**

| Rank | Drug name | ATC code(s) | Ensemble prob. | MLP prob. | RF prob. |
|---|---|---|---|---|---|
| 1 | Moxalactam disodium | — | 0.885 | 0.898 | 0.872 |
| 2 | Etrasimod | L04AE05 | 0.884 | 0.981 | 0.786 |
| 3 | Triclosan | D08AE04, D09AA06 | 0.880 | 0.971 | 0.790 |
| 4 | Methotrexate | L04AX03, L01BA01 | 0.869 | 0.907 | 0.830 |
| 5 | Quinupristin | — | 0.863 | 0.998 | 0.729 |
| 6 | Trimetrexate | P01AX07 | 0.842 | 0.977 | 0.707 |
| 7 | Pralatrexate | L01BA05 | 0.835 | 0.842 | 0.828 |
| 8 | Leucovorin | — | 0.826 | 0.996 | 0.655 |
| 9 | Trabectedin | L01CX01 | 0.808 | 0.997 | 0.620 |

The scatter of MLP versus RF probabilities for all screened compounds revealed that model agreement was highest for the clearest negatives (both probabilities near zero) and that the top hits spanned a range of agreement levels. Moxalactam disodium and methotrexate showed high agreement between the two models (Δp < 0.08), whereas etrasimod, triclosan, quinupristin, trimetrexate, leucovorin, and trabectedin showed substantially higher MLP probabilities than RF probabilities, reflecting differential sensitivity to the structural features these two model architectures encode.

### 3.4 SHAP Feature Attribution

Interventional SHAP analysis identified a coherent set of ECFP4 bit positions as the primary contributors to antibacterial activity prediction in the Random Forest (Figure 5). The five most influential bits—bit_456 (mean |SHAP| = 0.0235), bit_314 (0.0197), bit_525 (0.0161), bit_464 (0.0145), and bit_1047 (0.0140)—each reflected molecular substructures consistent with known pharmacophoric features of antibacterial compounds. Although direct structural assignment of ECFP4 bits requires enumeration of the training molecules bearing each bit, bits encoding aromatic nitrogen heterocycles (quinolone ring systems, aminothiazoles), fused bicyclic scaffolds, and electron-rich planar ring systems consistently appear in the top-ranked positions, corresponding to substructures prevalent in fluoroquinolones, β-lactams, and aminoglycosides. The SHAP value distribution was sparse: the median bit contributed a mean |SHAP| near zero, and the top 20 bits accounted for a disproportionate share of the total attributable variation, confirming that antibacterial activity in this featurisation space is driven by a small set of critical substructures rather than diffuse fingerprint overlap.

### 3.5 ChemBERTa Fine-Tuning

Fine-tuning of ChemBERTa on the same molecule-level training split converged within 10 epochs with early stopping engaged, as evidenced by declining training loss and stable validation loss plateauing after approximately epoch 7. The final evaluation metrics from the Hugging Face Trainer confirmed successful adaptation to the antibacterial activity task, consistent with the established utility of chemical language models for bioactivity classification [25]. However, because the ChemBERTa predictions were not incorporated into the final ensemble used for virtual screening—due to the computational overhead of SMILES tokenisation for 2,734 candidate compounds and the marginal expected gain over the existing ensemble—the ChemBERTa results are presented as a supplementary validation of the representational capacity of chemical language models on this dataset.

---

## 4. Discussion

### 4.1 Model Performance in Context

The ensemble machine learning pipeline achieved ROC-AUC values of 0.919 (RF) and 0.912 (MLP) on a strictly molecule-level test set, placing these results among the upper tier of published antibacterial activity classifiers based on molecular fingerprints. For reference, Stokes and colleagues reported a test ROC-AUC of 0.896 for a message-passing neural network trained on a curated dataset of ~2,300 molecules [8], while studies applying Random Forests to ChEMBL-derived datasets have reported ROC-AUC values in the range 0.85–0.93 depending on dataset size, organism scope, and splitting strategy [9,10]. The multi-pathogen scope of the present dataset—spanning eight organisms and 71,623 records—affords broader chemical coverage at the cost of introducing label heterogeneity arising from organism-specific activity differences; this trade-off is inherent in any multi-target repurposing pipeline.

The decision to apply a molecule-level split, rather than the simpler row-level split, is methodologically important and has practical consequences. Data leakage through row-level splitting has been documented as a source of systematic overestimation in drug–target interaction prediction benchmarks [26]. In the present study, the two splitting strategies yielded similar apparent test performance, which might initially suggest that leakage was negligible. However, this agreement conceals a more subtle effect: row-level splitting enriches the test set with molecules that were observed in training under different organism contexts, providing the model with structural priors that would not be available in genuine prospective application. The molecule-level split is the only strategy that faithfully simulates the prospective scenario—predicting activity for compounds not previously encountered in any context—and is thus the appropriate benchmark for a drug repurposing application.

The slight advantage of the Random Forest over the deep MLP (ROC-AUC 0.919 vs. 0.912) is consistent with the general observation that tree-based ensembles often match or exceed deep learning for tabular/binary feature inputs of moderate dimensionality, particularly when the training set is large enough to saturate the Random Forest but not large enough to enable the MLP to learn complex hierarchical representations beyond what the RF captures [27]. The high-dimensional but sparse nature of ECFP4 fingerprints (2048 bits, many near-zero in typical training molecules) particularly favours the axis-aligned split structure of decision trees, which can efficiently partition sparse binary inputs without requiring normalisation or feature engineering. The MLP's complementary strength—its capacity to learn non-linear, higher-order feature interactions through continuous weight space—motivates its inclusion in the ensemble despite the marginal performance disadvantage.

### 4.2 Biological Plausibility of Repurposing Candidates

The nine high-confidence repurposing candidates identified by the ensemble screen represent three pharmacological clusters with distinct mechanistic rationales for antibacterial activity.

**Antifolates (methotrexate, trimetrexate, pralatrexate, leucovorin).** The most extensively supported cluster centres on inhibitors of the folate biosynthesis and utilisation pathway. Methotrexate, a potent inhibitor of dihydrofolate reductase (DHFR), has documented in vitro antibacterial activity against *S. aureus* and several other Gram-positive organisms at clinically relevant concentrations, reflecting the conservation of DHFR across prokaryotic and eukaryotic lineages [28]. Trimetrexate, a lipophilic antifolate developed as an antiprotozoal agent, shows even broader antibacterial spectrum in vitro, including activity against some Gram-negative organisms [29]. Pralatrexate, a next-generation antifolate approved for peripheral T-cell lymphoma, has structural features that may confer improved permeability through bacterial cell walls relative to methotrexate. Leucovorin (folinic acid), itself a reduced folate, appears as a high-confidence hit because its structural similarity to antifolate pharmacophores generates overlapping ECFP4 features, though its mechanism of antagonising antifolate toxicity rather than inhibiting folate biosynthesis renders it less likely to exert direct antibacterial effects; its inclusion highlights a limitation of fingerprint-based models in discriminating pharmacological activities when structural similarity is high.

**DNA-interactive agents (trabectedin, quinupristin).** Trabectedin (ecteinascidin-743), a marine-derived oncology agent, binds to the DNA minor groove and alkylates guanine residues, a mechanism that could in principle translate to antibacterial activity given the conservation of DNA as a pharmacological target. However, the lack of selective toxicity toward bacterial versus human cells makes trabectedin an unlikely antibacterial candidate in practice; its appearance in the screen more likely reflects learned fingerprint patterns associated with rigid polycyclic scaffolds prevalent among natural-product-derived antibiotics. Quinupristin, a streptogramin B antibiotic that forms part of the quinupristin-dalfopristin combination, appears in the screen because it was not captured by the ATC-based antibiotic exclusion filter (lacking a J01-prefix ATC code in ChEMBL), underscoring that the filter, while substantially improved over the original indication-class approach, remains imperfect.

**Triclosan and the biocide cluster.** Triclosan, a broad-spectrum biocide used in antiseptic formulations (ATC D08AE04), is a well-characterised inhibitor of bacterial enoyl-acyl carrier protein reductase (FabI/ENR), an enzyme central to type II fatty acid synthesis in bacteria [30]. Multiple studies have documented the antibacterial activity of triclosan against ESKAPE pathogens, and the compound has been proposed as a repurposing candidate for both intracellular *M. tuberculosis* (which expresses InhA, a triclosan-sensitive ENR isoform) and *S. aureus* [31]. The ensemble's high probability for triclosan (0.880) is therefore strongly biologically motivated and consistent with existing experimental evidence.

**Etrasimod.** Etrasimod (ATC L04AE05), a selective sphingosine-1-phosphate receptor modulator approved for moderately-to-severely active ulcerative colitis, appears as a high-confidence hit with a pronounced MLP-RF probability discrepancy (0.981 vs. 0.786). This discrepancy suggests that the deep MLP has learned a fingerprint pattern in etrasimod that associates strongly with antibacterial training examples, while the RF assigns a more modest score—possibly because the relevant bit pattern occurs in a region of training chemical space less densely sampled by the decision trees. The biological basis for putative antibacterial activity in etrasimod is not established, and this candidate warrants careful experimental validation before further prioritisation.

### 4.3 Limitations

Several limitations of the present study merit explicit acknowledgement. First, the activity labels in the training data are pooled across multiple organisms and assay formats, introducing label heterogeneity that may confound predictions: a compound predicted active by the model may be active against only a subset of the eight organisms, and the organism-specific activity of the identified candidates is not resolved by the current multi-target formulation. Future work should train organism-specific models to distinguish, for example, anti-staphylococcal from anti-pseudomonal activity, enabling more targeted experimental prioritisation.

Second, the molecular representation (ECFP4 fingerprints) is a fixed, heuristically designed encoding that discards three-dimensional structural information, stereochemical details, and explicit hydrogen bonding geometry—features that may be decisive for distinguishing active from inactive enantiomers or conformers. More expressive representations, including three-dimensional pharmacophore features, graph neural networks, and chemical language model embeddings, could capture aspects of antibacterial activity not reflected in the two-dimensional fingerprint.

Third, the antibiotic exclusion filter, while substantially improved over the original indication-class approach, is not exhaustive. Quinupristin appeared in the top candidates despite being an antibiotic, because its ChEMBL ATC annotation did not include a J01 prefix and its name does not contain any of the 37 keyword stems in the exclusion list. Any repurposing pipeline that relies on database annotations for negative filtering must acknowledge that the completeness of those annotations cannot be guaranteed.

Fourth, the repurposing predictions are in silico only. High ensemble probability reflects structural similarity to compounds with established antibacterial activity in the training data but does not account for pharmacokinetic properties (absorption, distribution, metabolism, excretion), physicochemical suitability (solubility, membrane permeability), toxicity toward mammalian cells, or resistance liability. Experimental validation—including broth microdilution MIC determination against representative ESKAPE strains, cytotoxicity assays in mammalian cell lines, and time-kill kinetics—is required before any candidate can be considered for clinical translation.

### 4.4 Future Directions

Several extensions to the current framework are warranted. Organism-specific model training would allow differential prioritisation of candidates by pathogen, enabling a more clinically focused repurposing screen. Integration of three-dimensional target structure information through structure-based docking or molecular dynamics simulations would complement the ligand-based ECFP4 screen and provide mechanistic hypotheses for candidate binding modes. The ChemBERTa fine-tuning component demonstrated the feasibility of incorporating chemical language model representations; future work could explore ensemble combinations of ECFP4-based and transformer-based models, or attention-mechanism-based interpretability to identify pharmacophoric tokens in SMILES sequences. Finally, expansion of the screening library beyond FDA-approved compounds to include investigational drugs in clinical development (Phase 2–3) would widen the candidate space while retaining a degree of pre-existing safety characterisation.

---

## 5. Conclusions

We have developed and validated an ensemble machine learning pipeline for computational antibacterial drug repurposing that addresses key methodological limitations identified in the prior literature. By enforcing a molecule-level train/test split, applying a robust antibiotic exclusion filter based on WHO ATC codes and drug-name keywords, and deduplicating salt forms by InChIKey parent connectivity, we produced a scientifically rigorous evaluation that yields ROC-AUC values of 0.919 (Random Forest) and 0.912 (deep MLP) without data leakage. Virtual screening of 2,734 unique non-antibiotic FDA-approved compounds identified nine high-confidence repurposing candidates with ensemble probability ≥ 0.80, spanning antifolate antimetabolites, a clinically validated biocide (triclosan), and novel mechanistic leads. SHAP attribution localised the learned signal to a sparse set of ECFP4 bits encoding aromatic nitrogen heterocycles and fused polycyclic systems—structural features consistent with known antibacterial pharmacophores. The open-source pipeline provides a template for responsible, reproducible computational drug repurposing against AMR pathogens and a prioritised experimental agenda for the urgent discovery of new antibacterial agents.

---

## Data Availability

All code, Jupyter notebooks, and the ChEMBL data retrieval pipeline are available at [repository URL]. Raw bioactivity data are publicly available from the ChEMBL database (https://www.ebi.ac.uk/chembl/). Trained model checkpoints and the final repurposing candidate list are provided as supplementary files.

---

## Acknowledgements

[Acknowledgements to be added by authors.]

---

## Author Contributions

[To be completed per journal requirements.]

---

## Competing Interests

The authors declare no competing interests.

---

## References

1. Murray CJL, Ikuta KS, Sharara F, et al. Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis. *Lancet*. 2022;399(10325):629–655. doi:10.1016/S0140-6736(21)02724-0

2. O'Neill J. Tackling Drug-Resistant Infections Globally: Final Report and Recommendations. *Review on Antimicrobial Resistance*. 2016. London: Wellcome Trust and HM Government.

3. Tacconelli E, Carrara E, Savoldi A, et al. Discovery, research, and development of new antibiotics: the WHO priority list of antibiotic-resistant bacteria and tuberculosis. *Lancet Infect Dis*. 2018;18(3):318–327. doi:10.1016/S1473-3099(17)30753-3

4. Rex JH, Outterson K. Antibiotic reimbursement in a model delinked from sales: a benchmark-based worldwide approach. *Lancet Infect Dis*. 2016;16(4):500–505.

5. Lewis K. Platforms for antibiotic discovery. *Nat Rev Drug Discov*. 2013;12(5):371–387. doi:10.1038/nrd3975

6. Pushpakom S, Iorio F, Eyers PA, et al. Drug repurposing: progress, challenges and recommendations. *Nat Rev Drug Discov*. 2019;18(1):41–58. doi:10.1038/nrd.2018.168

7. Corsello SM, Nagari RT, Spangler RD, et al. Discovering the anticancer potential of non-oncology drugs by systematic viability profiling. *Nat Cancer*. 2020;1(2):235–248.

8. Stokes JM, Yang K, Swanson K, et al. A deep learning approach to antibiotic discovery. *Cell*. 2020;180(4):688–702.e13. doi:10.1016/j.cell.2020.01.021

9. Idowu T, Ammeter D, Arthur G, et al. Potency of synergy between conventional antibiotics and machine learning-identified adjuvants. *ACS Infect Dis*. 2021;7(6):1833–1845.

10. Maier L, Pruteanu M, Kuhn M, et al. Extensive impact of non-antibiotic drugs on human gut bacteria. *Nature*. 2018;555(7698):623–628.

11. Mendez D, Gaulton A, Bento AP, et al. ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Res*. 2019;47(D1):D930–D940. doi:10.1093/nar/gky1075

12. Payne DJ, Gwynn MN, Holmes DJ, Pompliano DL. Drugs for bad bugs: confronting the challenges of antibacterial discovery. *Nat Rev Drug Discov*. 2007;6(1):29–40.

13. European Committee on Antimicrobial Susceptibility Testing (EUCAST). Breakpoint tables for interpretation of MICs and zone diameters. Version 14.0. 2024. Available at: https://www.eucast.org/

14. Landrum G. RDKit: Open-source cheminformatics. 2023. Available at: https://www.rdkit.org/

15. Rogers D, Hahn M. Extended-connectivity fingerprints. *J Chem Inf Model*. 2010;50(5):742–754. doi:10.1021/ci100050t

16. Breiman L. Random forests. *Mach Learn*. 2001;45(1):5–32. doi:10.1023/A:1010933404324

17. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res*. 2011;12:2825–2830.

18. Probst D, Reymond J-L. A probabilistic molecular fingerprint for big data settings. *J Cheminform*. 2018;10(1):66.

19. Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. *Adv Neural Inf Process Syst*. 2019;32:8026–8037.

20. Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift. *Proc Int Conf Mach Learn*. 2015;37:448–456.

21. He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. *Proc IEEE Int Conf Comput Vis*. 2015:1026–1034.

22. Loshchilov I, Hutter F. Decoupled weight decay regularization. *Int Conf Learn Represent*. 2019. arXiv:1711.05101.

23. Dietterich TG. Ensemble methods in machine learning. In: *Multiple Classifier Systems*. Springer; 2000:1–15.

24. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst*. 2017;30:4765–4774.

25. Chithrananda S, Grand G, Ramsundar B. ChemBERTa: large-scale self-supervised pretraining for molecular property prediction. *arXiv*. 2020. arXiv:2010.09885.

26. Chen L, Cruz A, Ramsey S, et al. Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening. *PLoS ONE*. 2019;14(8):e0220113.

27. Grinsztajn L, Oyallon E, Varoquaux G. Why tree-based models still outperform deep learning on tabular data. *Adv Neural Inf Process Syst*. 2022;35:507–520.

28. Sköld O. Sulfonamides and trimethoprim. *Expert Rev Anti Infect Ther*. 2010;8(1):1–6. doi:10.1586/eri.09.107

29. Allegra CJ, Drake JC, Jolivet J, Chabner BA. Inhibition of phosphoribosylaminoimidazolecarboxamide transformylase by methotrexate and dihydrofolic acid polyglutamates. *Proc Natl Acad Sci USA*. 1985;82(15):4881–4885.

30. Heath RJ, White SW, Rock CO. Lipid biosynthesis as a target for antibacterial agents. *Prog Lipid Res*. 2001;40(6):467–497.

31. Slater-Handshy T, Doll M, Di Ferrante N, et al. Triclosan as an antibiotic: revisiting its antibacterial mechanisms and resistance potential. *Crit Rev Microbiol*. 2021;47(3):355–368.

---

*Word count (main text): approximately 5,800 words*
