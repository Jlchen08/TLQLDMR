Ms. Ref. No.: EGY-D-26-09344

Title: Transfer Learning Quantile Large-margin Distribution Machine for Wind Power Forecasting under Extreme Events

Energy

Dear Professor Wang,

The review of your paper is now complete, the Reviewers' reports are below. As you can see, the Reviewers present important points of criticism and a series of recommendations. We kindly ask you to consider all comments and revise the paper accordingly in order to respond fully and in detail to the Reviewers' recommendations. If this process is completed thoroughly, the paper will be acceptable for a second review.

If you choose to revise your manuscript it will be due into the Editorial Office by the Oct 18, 2026

Once you have revised the paper accordingly, please submit it together with a detailed description of your response to these comments. Please, also include a separate copy of the revised paper in which you have marked the revisions made.

Please note if a reviewer suggests you to cite specific literature, you should only do so if you feel the literature is relevant and will improve your paper. Otherwise please ignore such suggestions and indicate this fact to the handling editor in your rebuttal.

To submit a revision, please go to https://www.editorialmanager.com/egy/ and login as an Author.

Your username is: ********

If you need to retrieve password details, please go to: ********.

NOTE: Upon submitting your revised manuscript, please upload the source files for your article. For additional details regarding acceptable file formats, please refer to the Guide for Authors at: http://www.elsevier.com/journals/energy/0360-5442/guide-for-authors

When submitting your revised paper, we ask that you include the following items:

Manuscript and Figure Source Files (mandatory):
We cannot accommodate PDF manuscript files for production purposes. We also ask that when submitting your revision you follow the journal formatting guidelines. Figures and tables may be embedded within the source file for the submission as long as they are of sufficient resolution for Production. For any figure that cannot be embedded within the source file (such as *.PSD Photoshop files), the original figure needs to be uploaded separately. Refer to the Guide for Authors for additional information.
http://www.elsevier.com/journals/energy/0360-5442/guide-for-authors

Highlights (mandatory):
Highlights consist of a short collection of bullet points that convey the core findings of the article and should be submitted in a separate file in the online submission system. Please use 'Highlights' in the file name and include 3 to 5 bullet points (maximum 85 characters, including spaces, per bullet point). See the following website for more information
http://www.elsevier.com/highlights

Thank you very much for expressing your interest in ENERGY.

Sincerely,

Diangui Huang
Subject Editor
Energy


Reviewers’ comments on the manuscript:

Reviewer #1: 1- The manuscript's novelty is promising, but the methodological contribution is not yet isolated clearly enough from existing SVQR, LDMR, and transfer-learning frameworks. The paper should provide a sharper theoretical comparison showing exactly what is new in TL-QLDMR beyond combining known components, and why the proposed combination is not simply an engineering integration of established ideas.
2- The definition of "extreme events" is somewhat heuristic and may bias the target-domain construction. The use of 98th-percentile thresholds, temperature cutoffs, and a fixed 25 m/s cut-out rule should be justified more rigorously, and the authors should include a sensitivity analysis showing how results change under alternative extreme-event definitions.
3- The study evaluates the final model mainly on a single selected target farm, which limits the generality of the conclusions. A stronger experimental design would test the method through leave-one-farm-out or cross-farm transfer validation to demonstrate that the reported gains are not specific to Farm 6 alone.
4- The transfer-learning setting appears to rely on a very large source domain and a much smaller target domain, but the impact of target-label scarcity is not studied systematically. The authors should vary the number of target labels and report how performance degrades under stricter few-shot conditions, since this is central to the claimed extreme-event scenario.
5- The asymmetric variance regularizer is an interesting idea, but its mathematical justification remains incomplete. The manuscript should provide a more rigorous derivation explaining why the proposed asymmetric form is the most appropriate quantile-compatible modification of LDMR, and whether the stated positive-definiteness and convexity claims always hold under all kernel and hyperparameter settings.
6- The probabilistic evaluation is not sufficiently comprehensive for a quantile-forecasting paper. Reporting only a 95% interval setting is too narrow; the authors should evaluate multiple quantile levels, calibration curves, and ideally a proper scoring rule beyond Winkler/CWC to demonstrate stable probabilistic behavior across the distribution.
7- The comparison protocol may not be fully fair unless all baselines are tuned with equal care. Because the proposed method combines transfer learning, kernel alignment, and weighted quantile loss, the paper should document hyperparameter tuning budgets and validation procedures for every baseline, and include statistical significance tests over repeated runs.
8- The temporal validation strategy is not sufficiently explicit for a forecasting problem. The manuscript should clarify whether the train/validation/test split is strictly chronological; if random splitting was used anywhere, it may leak future information and overstate performance in a time-series setting.
9- The scalability claim is positive, but the computational analysis is incomplete. The Nyström-SGD solver should be evaluated more thoroughly across a wider range of landmark sizes and data scales, with an explicit accuracy-runtime trade-off curve, because the current runtime advantage alone does not show whether approximation error becomes significant.

10- The conclusions would be stronger if the manuscript tested robustness beyond injected noise on one farm. The paper should include ablations under multiple meteorological seasons, different extreme-event types, and possibly additional datasets, to show that the observed robustness is general rather than dataset-specific.
11- The literature review section should be significantly expanded and structured more critically, use these paper for example "Multiscale Wind Forecasting Using Explainable-Adaptive Hybrid Deep Learning", "Prediction of uncertainty ramping demand in new power systems based on a CNN-LSTM hybrid neural network", "Wind power forecasting enhancement utilizing adaptive quantile function and cnn-lstm: A probabilistic approach", "A multi-level model for hybrid short term wind forecasting based on SVM, wavelet transform and feature selection".



Reviewer #2: The manuscript proposes Transfer Learning Quantile Large-margin Distribution Machine for Regression (TL-QLDMR) for wind power forecasting under extreme operating conditions. The topic is timely and relevant to Energy. Forecasting under rare, high-risk wind-power regimes is important for renewable-energy integration and grid operation, and the attempt to combine transfer learning, quantile regression, large-margin distribution learning, and a scalable Nystrom-SGD solver is technically interesting.

Several points need clarification and additional evidence before the main claims can be fully assessed, especially the experimental scope, data-splitting protocol, baseline reproducibility, and the strength of the theoretical and deployment-related claims.

Major comments:

1. The experimental scope is narrower than the claims. The manuscript uses six wind farms to define and rank domain shift, but the main deterministic prediction comparison, probabilistic interval evaluation, noise robustness test, ablation study, residual-density analysis, and scalability study are all reported on Farm 6. Since Farm 6 is selected as the farm with the largest composite shift, this is better described as a selected stress-test case, not as full six-farm validation. The authors should either report the main comparisons across all six farms, or at least representative high-, medium-, and low-shift farms, or clearly narrow the claims to a Farm 6 evaluation after six-farm screening.

2. The data split and target-domain partition are insufficiently described. For Farm 6, the paper reports 66,009 source-domain normal samples and 4,144 target-domain extreme-event samples, but it does not state how many target samples are used for training, validation, calibration, and testing. This is critical because the proposed method uses target-domain labels through weighted pinball loss. The authors should report the exact chronological span, forecasting horizon, source/target train-validation-test counts, split strategy, and whether thresholds, normalization, calibration, and hyperparameter tuning are based only on training data.

3. The time-series leakage risk should be addressed. A fixed random seed does not by itself make a valid forecasting protocol. Random splits may place neighboring 15-minute samples from the same event into both training and test sets. The authors should use a chronological, blocked, rolling-origin, or event-level split, or at least compare the current protocol with a leakage-safe chronological protocol.

4. The extreme-event definition needs stronger validation. The target domain is defined by a union of high wind speed, low/high temperature, and wind speed above 25 m/s rules. The authors should report per-rule sample counts, overlaps, event durations, seasonal distribution, and representative episodes. They should also justify the exclusion of ramp events, since rapid ramps are important wind-power operational risk cases, or clearly state that ramp-risk forecasting is outside the paper's scope.

5. The method derivation needs clarification. The asymmetric variance regularizer is central to the paper, but the current derivation is difficult to verify. The authors should explain what quantities are treated as constants, why this is mathematically valid when residuals depend on the prediction function, and whether the variance term is computed on the source domain, target domain, or combined domain. The relation between the exact kernel formulation, the Nystrom approximation, and the SGD optimizer should also be separated more clearly.

6. The theoretical claims are overstated. The manuscript refers to positive definiteness, convergence, and theoretical guarantees, but the proof conditions are only sketched. For a nonsmooth weighted pinball-loss objective with Nystrom approximation and SGD, the authors should specify assumptions on kernel definiteness, landmark selection, step-size schedule, regularization, bounded gradients, and approximation error. Otherwise, the claims should be softened.

7. Baseline fairness and reproducibility are not sufficient. The paper compares TL-QLDMR with many baselines, but does not provide enough information about implementation sources, hyperparameter search spaces, selected hyperparameters, validation criteria, calibration procedures, random seeds, or repeated runs. The authors should also report confidence intervals or standard deviations and statistical tests against the strongest baselines. This is important because some improvements are modest, for example the RMSE improvement over HHO-SVR in Table 4.

8. The probabilistic forecasting evaluation should be strengthened. A single 95% interval table is not enough to establish interval quality. The authors should provide calibration curves or reliability diagrams, empirical coverage at multiple nominal levels, and sharpness-coverage trade-off results. They should also clarify whether TL-QLDMR is evaluated directly from quantile outputs or receives the same post-hoc calibration as conformal baselines.

9. The noise-robustness experiment needs more detail. The authors should state which variables receive Gaussian noise, whether noise is added before or after normalization, whether wind direction is handled as a circular variable, whether the target power output is perturbed, and whether results are averaged over multiple noise realizations. The result that TL-QLDMR has higher R2 at SNR = 60 dB than in the clean setting also needs explanation.

10. The real-time and scalability claims should be restrained. The reported training time on a server with eight NVIDIA Tesla V100 GPUs is useful, but it does not by itself demonstrate real-time wind-farm deployment. The authors should report inference latency, memory footprint, CPU or single-GPU performance, online update cost, and the cost of landmark selection and preprocessing. If real-time operation is claimed, the operational latency requirement should be defined.

11. The interpretability claim is not demonstrated. Kernel models may be more structured than some deep networks, but the manuscript does not provide feature attribution, sensitivity analysis, physical consistency checks, or operational interpretability examples. The interpretability language should be supported with evidence or made more cautious.

12. The data and code availability statement is too weak. For a method paper with custom extreme-event labels, many baselines, custom splits, and a new solver, "available upon reasonable request" is not sufficient. The authors should provide code, preprocessing scripts, exact source/target labels, train/validation/test split files, baseline configuration files, scripts for generating the main tables and figures, random seeds, and environment details.

Minor comments:

1. The abstract is dense and overstates the current evidence. It should distinguish the proposed method, the Farm 6 stress-test results, and the additional validation needed for broader deployment.

2. The Highlights should be revised. The statement that "Six-farm experiments show better accuracy, interval quality, and scalability" is misleading because the main predictive comparisons are not reported across all six farms.

3. The related work section should engage more deeply with wind-power probabilistic forecasting, extreme-event forecasting, transfer learning in renewable-energy forecasting, conformal prediction, and recent deep spatiotemporal baselines.

4. Some baseline choices need justification. For example, ARA-SVR appears to come from traffic-flow forecasting rather than wind-power forecasting.

5. The input-window construction should be described more clearly, including lag variables, forecasting horizon, normalization, missing-value handling, and whether future meteorological variables are used.

6. Table 3 should include the proportion of extreme-event samples for each farm, not only absolute counts.

7. Figures 3-7 use selected temporal segments. The selection criterion should be stated to avoid the impression of cherry-picked examples.

8. The residual-density comparison should include quantitative residual statistics, such as skewness, kurtosis, tail-error frequency, and calibration error.

9. The formulas for CWC and Winkler score should be provided, including penalty terms and normalization.

10. A notation table would improve readability, as the manuscript introduces many symbols.

11. The term "extreme events" should be used consistently. The current definition is sample-level out-of-distribution operating states, not necessarily named or tracked meteorological events.

12. The conclusion should include limitations, including the single-farm main evaluation, sample-level extreme-event definition, need for chronological validation, and lack of deployment-latency evidence.

Overall, the paper has a promising technical direction and addresses an important application. The manuscript would be substantially strengthened by broader experimental validation, a leakage-safe and fully specified data protocol, reproducible baseline comparisons, clearer mathematical justification, and more cautious claims about scalability, interpretability, and theoretical guarantees.


Reviewer #3: This manuscript focuses on the challenges of limited samples, cross-domain distribution differences, and uncertainty quantification in wind power forecasting under extreme events. It proposes a TL-QLDMR framework integrating large-margin distribution learning, quantile regression, domain adaptation, and scalable optimization. The topic is practically relevant, the overall technical framework is reasonably complete, and the experiments cover point forecasting, interval forecasting, ablation studies, and computational efficiency. To further improve the clarity, reliability, and reproducibility of the manuscript, the authors are encouraged to clarify several methodological definitions and derivations, provide more details on the data partitioning and experimental settings, and conduct additional validation where necessary.
1. The review of the existing literature on wind power forecasting under extreme events remains insufficient in the Introduction. After explaining the importance of extreme-event forecasting, the manuscript moves rather quickly to the general advantages of kernel methods and LDMR, without systematically discussing recent developments and limitations of deep learning, transfer learning, and probabilistic forecasting methods in extreme-event scenarios. Consequently, the motivation for adopting a kernel-based approach is not yet sufficiently clear. The authors are encouraged to expand the review of extreme-event wind power forecasting and explain the suitability, advantages, and potential trade-offs of kernel methods compared with neural networks or other learnable representation methods. This would strengthen the logical connection between the research problem and the proposed technical approach.
2. Please clarify how the source-domain mean residual in Eq. (4) is handled. According to its definition, this quantity is calculated from the model residuals and therefore appears to depend on the model parameters being optimized. However, the manuscript states that it is treated as a scalar constant during optimization. Meanwhile, the centered residual representation in Eq. (7) appears to imply that the mean residual is calculated from the current model. Please clarify whether it is calculated before optimization and then fixed, or updated during training as the model parameters change. If it is fixed, its calculation and initialization procedure should be explained. If it is updated dynamically, the expression "scalar constant" should be revised, and consistency among Eqs. (4), (6)-(7), and the subsequent kernelized derivation should be ensured.
3. Please clarify the definition and derivation of the MMD-based domain adaptation term. In Eq. (4), it is presented as the standard distance between the source- and target-domain mean embeddings under a fixed RKHS mapping. When the kernel mapping is fixed, this formulation appears to be independent of the model parameters being optimized. However, Eq. (10) expresses it as a kernelized quadratic term involving the model coefficients, which appears to measure the difference between the mean predictions or projected means of the two domains. The authors should explain how Eq. (4) is transformed into Eq. (10) and clarify whether this term aligns the complete feature-space distributions or only constrains the difference between the two domains along the prediction direction.
4. The mathematical basis of the proposed "asymmetric variance regularization term" should be clarified. The variance term in Eq. (4) is essentially based on squared centered residuals. Such a term penalizes positive and negative residuals in the same manner and does not explicitly depend on the quantile level. Therefore, its asymmetry and its essential difference from conventional symmetric variance regularization are not evident from the current formulation. The quantile-dependent asymmetry of the model appears to arise mainly from the pinball loss. If this is the case, the corresponding claims and descriptions of the variance regularization term should be revised accordingly.
5. The manuscript assigns a substantially larger penalty weight to the target domain than to the source domain in order to prevent the large number of source-domain samples from dominating joint training. However, the current objective function directly sums the losses from the two domains. Their relative contributions therefore depend not only on the selected penalty weights but also on the corresponding sample sizes. For example, for Farm 6, the source domain contains 66,009 samples, whereas the target domain contains 4,144 samples, resulting in an approximate sample-size ratio of 16 to 1. At the same time, the candidate target-domain penalty weights are much larger than the candidate source-domain weights, resulting in a per-sample weight ratio ranging from 250 to 20,000. Therefore, this setting may not merely compensate for sample imbalance but may cause the optimization to be dominated by the target-domain loss. The existing "w/o Transfer" and "w/o MMD" ablation studies cannot independently identify the contribution of this weighting mechanism. The authors are encouraged to report the final source- and target-domain penalty weights used in each experiment and conduct a sensitivity analysis of the weighting strategy. Possible comparisons could include equal weighting, domain-size-normalized weighting, sample-imbalance-based weighting, and the current strategy.
6. Please provide further details on how the training, validation, and test sets were constructed. Section 4.3 states that the data were randomly divided using a fixed random seed, while Figure 1 shows that the input samples were generated using sliding windows of length 24. Please clarify whether the datasets were divided chronologically, by independent time periods, or by randomly splitting the window samples. If overlapping windows can appear in different subsets, the authors should also explain how potential temporal information leakage was avoided.
Minor comment:
The authors are advised to check whether the Highlights comply with the journal's formatting and writing requirements, including the use of abbreviations and the character limit for each bullet point.


***
We invite you to convert your supplementary data (or a part of it) into an additional journal publication in Data in Brief, a multi-disciplinary open access journal. Data in Brief articles are a fantastic way to describe supplementary data and associated metadata, or full raw datasets deposited in an external repository, which are otherwise unnoticed. A Data in Brief article (which will be reviewed, formatted, indexed, and given a DOI) will make your data easier to find, reproduce, and cite.

 

You can submit to Data in Brief when you upload your revised manuscript. To do so, complete the template and follow the co-submission instructions found here: www.elsevier.com/dib-template. If your manuscript is accepted, your Data in Brief submission will automatically be transferred to Data in Brief for editorial review and publication.

 

Please note: an open access Article Publication Charge (APC) is payable by the author or research funder to cover the costs associated with publication in Data in Brief and ensure your data article is immediately and permanently free to access by all. For the current APC see: www.elsevier.com/journals/data-in-brief/2352-3409/open-access-journal

 

Please contact the Data in Brief editorial office at dib-me@elsevier.com or visit the Data in Brief homepage (www.journals.elsevier.com/data-in-brief/) if you have questions or need further information.



We invite you to submit a method article alongside your research article. This is an opportunity to get full credit for the time and money spent on developing research methods, and to increase the visibility and impact of your work. If your research article is accepted, we will contact you with instructions on the submission process for your method article to MethodsX. On receipt at MethodsX it will be editorially reviewed and, upon acceptance, published as a separate method article. Your articles will be linked on ScienceDirect.

Please prepare your paper using the MethodsX Guide for Authors: https://www.elsevier.com/journals/methodsx/2215-0161/guide-for-authors (and template available here: https://www.elsevier.com/MethodsX-template) Open access fees apply.

For further assistance, please visit our customer support site at http://help.elsevier.com/app/answers/list/p/7923 Here you can search for solutions on a range of topics, find answers to frequently asked questions and learn more about EM via interactive tutorials. You will also find our 24/7 support contact details should you need any further assistance from one of our customer support representatives.

At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_EGY#

To ensure this email reaches the intended recipient, please do not delete the above code

