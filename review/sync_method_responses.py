#!/usr/bin/env python3
"""Generate self-contained response entries from verified manuscript excerpts.

Original excerpts are checked against TLQLDMR_manuscript; revised excerpts are
checked against files actually included in the submitted TLQLDMR_R1 manuscript
(or an explicitly identified separate submission artifact). Deletions reproduce
the original passage and an explanation, never repository-only archive prose.
Only typesetting is adapted in the letter. Compile first to resolve refs.
"""
from pathlib import Path
import argparse,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'review/method_revision_changes.json'
BASE=ROOT/'TLQLDMR_response_1st/main.tex'

def normalize(s):
    return re.sub(r'\s+',' ',s).strip()

def load_refs(path):
    return dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',path.read_text()))

def submitted_sources():
    """Follow manuscript inputs, excluding commented-out repository archives."""
    entry=ROOT/'TLQLDMR_R1/TLQLDMR_energy.tex'
    pending=[entry];sources={}
    while pending:
        path=pending.pop().resolve()
        if path in sources:continue
        source=re.sub(r'(?m)(?<!\\)%.*$','',path.read_text())
        sources[path]=source
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',source):
            candidates=[base/name for base in (entry.parent,path.parent)]
            candidates=[p if p.suffix else p.with_suffix('.tex') for p in candidates]
            child=next((p for p in candidates if p.exists()),None)
            if child is None:raise ValueError('Missing submitted input: '+name)
            pending.append(child)
    return sources

def validate_records(records):
    sources=submitted_sources();errors=[]
    submitted=' '.join(normalize(s) for s in sources.values())
    for key,record in records.items():
        kind=record.get('change_type','replacement')
        if kind not in {'replacement','addition','deletion','relocation'}:
            errors.append('Unknown change type: '+key);continue
        original=record.get('original','');revised=record.get('revised','')
        ot=(ROOT/record.get('original_file','TLQLDMR_manuscript/TLQLDMR_energy.tex')).read_text()
        if kind=='addition':
            if original or not record.get('addition_note'):
                errors.append('Addition must identify absence of original text: '+key)
        elif not original or normalize(original) not in normalize(ot):
            errors.append('Original not literal: '+key)
        if kind=='deletion':
            if revised or not record.get('deletion_note'):
                errors.append('Deletion requires an empty replacement and explanation: '+key)
            if original and normalize(original) in submitted:
                errors.append('Deleted original still appears in submitted manuscript: '+key)
            archive=record.get('archive_file')
            if archive and (ROOT/archive).resolve() in sources:
                errors.append('Research archive is still included in submission: '+key)
            continue
        if kind=='relocation' and not record.get('relocation_note'):
            errors.append('Relocation must describe the submitted destination: '+key)
        file=record.get('file')
        if not file:
            errors.append('Non-deletion has no revised source: '+key);continue
        path=(ROOT/file).resolve()
        if not path.exists():
            errors.append('Revised source missing: '+key);continue
        if not revised or normalize(revised) not in normalize(path.read_text()):
            errors.append('Revised not literal: '+key)
        separate_highlights=(record.get('submission_role')=='separate_highlights'
                             and path==ROOT/'TLQLDMR_R1/Highlights.tex')
        if path not in sources and not separate_highlights:
            errors.append('Revised passage is outside the submitted manuscript: '+key)
    if errors:raise ValueError('\n'.join(errors))

def _source_position(source,needle):
    """Return a useful source offset even when only whitespace changed."""
    position=source.find(needle)
    if position>=0:return position
    words=normalize(needle).split()
    if not words:return 0
    # A short literal anchor is enough to identify the surrounding heading;
    # the complete excerpt is validated independently by validate_records().
    for length in (18,12,8,5):
        if len(words)<length:continue
        pattern=r'\s+'.join(re.escape(word) for word in words[:length])
        match=re.search(pattern,source)
        if match:return match.start()
    return 0

def _heading_chain(path,needle):
    if not path.exists():return []
    source=path.read_text()
    position=_source_position(source,needle)
    pattern=re.compile(r'\\(section|subsection|subsubsection)\*?\{((?:[^{}]|\{[^{}]*\})*)\}')
    latest={}
    for match in pattern.finditer(source):
        if match.start()>position:break
        level,title=match.groups()
        latest[level]=title.strip()
        if level=='section':
            latest.pop('subsection',None);latest.pop('subsubsection',None)
        elif level=='subsection':latest.pop('subsubsection',None)
    return [(level,latest[level]) for level in ('section','subsection','subsubsection') if level in latest]

def response_location(record):
    """Return a page-independent, reviewer-facing approximate location."""
    explicit=record.get('response_location')
    if explicit:return explicit.rstrip(' :')
    kind=record.get('change_type','replacement')
    if kind=='deletion':
        relative=record.get('original_file','TLQLDMR_manuscript/TLQLDMR_energy.tex')
        needle=record.get('original','')
    else:
        relative=record.get('file','')
        needle=record.get('revised','')
    path=(ROOT/relative).resolve() if relative else ROOT
    chain=_heading_chain(path,needle)
    filename=Path(relative).name
    if not chain and filename=='TLQLDMR_energy.tex':
        source=path.read_text()
        position=_source_position(source,needle)
        abstract_start=source.rfind(r'\begin{abstract}',0,position+1)
        abstract_end=source.rfind(r'\end{abstract}',0,position+1)
        if abstract_start>abstract_end:
            detail='near the corresponding passage in the original manuscript' if kind=='deletion' else 'near the revised passage'
            return 'Abstract, '+detail
    # Included files inherit their parent section, which is not repeated in
    # the input file itself.
    parent_sections={
        'chronological_study.tex':'Experiments',
        'selection_study.tex':'Experiments',
        'selection_tables.tex':'Experiments',
        'solver_accuracy.tex':'Proposed Method',
        'selection_appendix_metrics.tex':'Validation-selected comparison details',
        'selection_configs.tex':'Validation-selected comparison details',
        'framework.tikz':'Proposed Method',
        'Highlights.tex':'separate Highlights file',
    }
    if not any(level=='section' for level,_ in chain) and filename in parent_sections:
        chain.insert(0,('section',parent_sections[filename]))
    if filename=='selection_tables.tex' and not any(level=='subsection' for level,_ in chain):
        chain.append(('subsection','Validation-selected predictive comparison'))
    if filename=='framework.tikz':
        chain.append(('subsection','TL-QLDMR Model Formulation'))
    if filename=='Highlights.tex':return 'Separate Highlights file, newly prepared text'
    labels={'section':'Section','subsection':'subsection','subsubsection':'under the subheading'}
    parts=[labels[level]+' ``'+title+"''" for level,title in chain]
    if not parts:
        parts=['Revised manuscript, relevant passage']
    excerpt=needle.lstrip()
    if excerpt.startswith(r'\begin{table') or excerpt.startswith(r'\caption'):
        detail='near the corresponding table'
    elif excerpt.startswith(r'\begin{figure') or filename=='framework.tikz':
        detail='in the corresponding figure'
    elif kind=='deletion':detail='near the corresponding passage in the original manuscript'
    elif kind=='addition':detail='near the added passage'
    else:detail='near the revised passage'
    return ', '.join(parts+[detail])

def _note(text):
    return '\\begin{quote}\n\\small\n'+text.strip()+'\n\\end{quote}\n'

def render_change(key,record,refs):
    location=response_location(record)
    body='% BEGIN VERIFIED CHANGE '+key+'\n'
    body+=r'\noindent\textbf{$\blacktriangleright\blacktriangleright$ '+location+r':}\par'+'\n\n'
    body+=r'\noindent$\longrightarrow$\enspace\textbf{Original text:}\par'+'\n'
    if record.get('change_type')=='addition':body+=_note(record['addition_note'])
    else:body+=quote(record['original'],'original',refs['original'])
    body+='\n'+r'\noindent$\longrightarrow$\enspace\textbf{Revised text:}\par'+'\n'
    if record.get('change_type')=='deletion':
        body+=_note(record['deletion_note'])
    else:
        if record.get('relocation_note'):body+=_note(record['relocation_note'])
        body+=quote(record['revised'],'revised',refs['revised'])
    return body+'% END VERIFIED CHANGE '+key+'\n\n'

def quote(text,side,refs):
    # Preserve every quoted word/formula; avoid importing manuscript labels.
    # The CAS title command's optional argument is not understood by IEEEtran;
    # show the title text itself in the response instead of printing the option.
    text=re.sub(r'\\title\s*(?:\[[^]]*\])?\{([^{}]*)\}',r'\1',text)
    # Tables already receive a non-expanding width limit below. Remove an
    # outer manuscript resizebox before applying that quote-specific layout.
    text=re.sub(r'\\resizebox\{\\(?:line|text)width\}\{!\}\{%?\s*(\\begin\{tabular\}[\s\S]*?\\end\{tabular\})\}',r'\1',text)
    text=re.sub(r'(?m)^[ \t]*\\label\{[^}]+\}[ \t]*\n','',text)
    text=re.sub(r'\\label\{[^}]+\}','',text)
    # Included result tables are reproduced by their own verified records.
    # Do not make the response build depend on the manuscript's input path.
    text=re.sub(r'(?m)^[ \t]*\\input\{[^}]+\}[ \t]*$', '', text)
    def resolve(m):
        name=m.group(2)
        if name not in refs:raise ValueError('Unresolved manuscript reference: '+name)
        return '('+refs[name]+')' if m.group(1)=='eqref' else refs[name]
    text=re.sub(r'\\(eqref|ref)\{([^}]+)\}',resolve,text)
    text=text.replace(r'\begin{equation}',r'\begin{equation*}').replace(r'\end{equation}',r'\end{equation*}')
    text=text.replace(r'\begin{remark}','').replace(r'\end{remark}','')
    text=re.sub(r'\\(?:subsection|subsubsection)\{([^}]+)\}',r'\\textbf{\1}',text)
    text=re.sub(r'\\begin\{table\*?\}(?:\[[^\]]*\])?',r'\\begingroup',text)
    text=re.sub(r'\\end\{table\*?\}',r'\\endgroup',text)
    text=re.sub(r'\\begin\{figure\*?\}(?:\[[^\]]*\])?',r'\\par\\begingroup',text)
    text=re.sub(r'\\end\{figure\*?\}',r'\\par\\endgroup',text)
    text=re.sub(r'\\captionof\{figure\}\{',r'\\par\\noindent\\textit{',text)
    text=text.replace(r'\caption{',r'\par\noindent\textit{')
    text=re.sub(r'\\captionsetup\{[^}]+\}','',text)
    text=text.replace(r'\FloatBarrier','')
    text=text.replace(r'\begin{algorithm}[t]',r'\begin{minipage}{\linewidth}'+ '\n'+r'\begin{algorithm}[H]')
    text=text.replace(r'\end{algorithm}',r'\end{algorithm}'+ '\n'+r'\end{minipage}')
    text=text.replace(r'\begin{tabular}',r'\par\begin{adjustbox}{max width=\linewidth}%'+'\n'+r'\begin{tabular}')
    text=text.replace(r'\end{tabular}',r'\end{tabular}'+'\n'+r'\end{adjustbox}\par')
    text=re.sub(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}',lambda m:r'\includegraphics[width=0.65\linewidth]{../'+('TLQLDMR_manuscript' if side=='original' else 'TLQLDMR_R1')+'/'+m.group(2)+'}',text)
    if text.lstrip().startswith(r'\item'):
        text='\\begin{itemize}\n'+text+'\n\\end{itemize}'
    # The small code-native figure labels contain deliberate line breaks.
    if '\\begin{' not in text and '\\\\' in text:
        text=text.replace('\\\\',r'\newline ')
    text='\n'.join(line.rstrip() for line in text.splitlines())
    return '\\begin{quote}\n\\small\n'+text.strip()+'\n\\end{quote}\n'

# Each response is independently complete; no referrals to another comment.
RESPONSES={
'Reviewer 1, Concern 1':(
 r'Thank you for asking us to distinguish the contribution from its established components. We now describe a specified supervised-transfer formulation rather than a new asymmetric variance: normalized centered source-residual variance controls dispersion, weighted pinball loss supplies quantile-dependent underprediction/overprediction costs, and a projected mean-discrepancy term constrains the predictive direction. The revision gives a consistent kernel derivation and explicit-feature counterpart, including the label-dependent linear term. It does not claim that this combination is uniquely optimal or that variance minimization forces a symmetric residual distribution. The earlier empirical records are distinguished from validation of this corrected objective.',
 ['intro_framework','contributions','related_variance','variance_roles','domain_roles','weighting','experiment_scope']),
'Reviewer 1, Concern 3':(
 r'We agree that six-farm screening does not establish cross-farm predictive generalization. The Abstract and contributions now distinguish domain screening on six farms from selection of Farm~6 as the stress-test case. The archived predictions are also identified as earlier implementation results, rather than evidence for the corrected objective. No leave-one-farm-out or cross-farm validation of the corrected model is claimed; such validation remains outstanding.',
 ['abstract','contributions','limitations']),
'Reviewer 1, Concern 4':(
 r'The requested target-label scarcity analysis remains necessary. We have withdrawn the earlier few-shot performance claim because the solver version underlying those scores has not been established and those scores cannot be assigned to the corrected residual-variance objective. The manuscript now specifies a fixed-test, explicit-count, test-independent protocol instead of presenting unverified results as a completed sensitivity study. The supplied Farm~6 inputs now support a fixed-configuration four-variant diagnostic, but that run does not vary the target-label budget; corrected-model scarcity results therefore remain outstanding. We therefore do not claim that this empirical request has been completed.',
 ['fewshot_evidence','data_partition']),
'Reviewer 1, Concern 5':(
 r'Thank you for identifying the need to justify the regularizer and state the limits of the theory. The variance penalty is symmetric and contains no $\tau$; we have removed the claim that it is a new asymmetric variance or the most appropriate quantile-specific form. Its motivation is the translation invariance $\|H_S(u+a\mathbf1)\|^2=\|H_Su\|^2$, which permits nonzero residual means while penalizing dispersion. Quantile dependence enters through weighted pinball loss, not a prescribed residual skewness. We also remove the claim that variance minimization forces symmetry or Gaussian errors. With a positive-semidefinite kernel, nonnegative regularization/loss weights, and $\tau\in(0,1)$, the objective is jointly convex. The proof uses $S\succeq0$, $M=aa^T\succeq0$, and the quadratic form of $\Omega$. If $K\succ0$, $\Omega\succ0$ and strong convexity holds in $\beta$ for fixed $b$; the unregularized intercept prevents a joint strong-convexity claim. Singular kernels use the Moore--Penrose inverse and need not have unique coefficients, although null-space coefficients represent the same predictor. The Nystr\"{o}m matrix similarly satisfies $R\succeq I$ only for the stated nonnegative penalties. We make no Adam convergence-rate or exact-kernel approximation-error claim. The former SymVar comparison is excluded because it neither represents a distinct symmetric residual-variance ablation of this model nor has consistent, reproducible component results. The new four-variant fixed-configuration run gives full-model RMSE 7.531988 MW and no-variance RMSE 7.531987 MW; the difference is below $10^{-6}$ MW and does not demonstrate a variance benefit. Full-model calibrated PICP is 0.971084 and MPIW is 27.698317 MW. All variants and their saved predictions are reported. This historical random-split diagnostic does not establish chronological generalization or statistical significance.',
 ['abstract','intro_challenges','variance_roles','objective','variance_matrix','convexity','beta_recovery','dual_matrices','nystrom_R','nystrom_c','convergence','ablation','conclusion_method']),
'Reviewer 1, Concern 10':(
 r'We agree that one farm and injected perturbations do not establish robustness across seasons, event types, or datasets. The revised Abstract, experimental scope, and Conclusion now limit the evidence accordingly. They also state that archived prediction and runtime records concern earlier implementations and cannot establish the behavior of the corrected objective. No new seasonal, cross-farm, or external-dataset result is claimed.',
 ['abstract','experiment_scope','legacy_point_noise','legacy_interval_noise','limitations']),
'Reviewer 2, Major Comment 1':(
 r'The distinction between screening and predictive evaluation is now explicit. Six farms support domain screening, while Farm~6 is the selected stress-test case. We have additionally separated archived implementation scores from validation of the corrected residual-variance objective; a new fixed-configuration Farm~6 diagnostic now evaluates the corrected model, while chronological and cross-farm predictive validation remain outstanding. The Abstract and Conclusion no longer assert six-farm predictive validation or deployment readiness.',
 ['abstract','contributions','experiment_scope','limitations']),
'Reviewer 2, Major Comment 2':(
 r'The archived Farm~6 window dataset contains 66,009 source samples and 4,143 target samples; the seed-0 random 90/10 target split gives 3,728 training and 415 test windows. We have removed the unsupported statement that all historical thresholds and preprocessing used training data only: the historical meteorological thresholds used the full record, and random windows are not a chronological holdout. The corrected validation protocol reserves calibration labels from target training and fits preprocessing only on model-training data. The available archive does not establish every historical validation/calibration assignment or the full requested chronological protocol, so the supplied data do not recover those historical choices. We now report the actual diagnostic counts: point fitting uses 3,728 target labels; interval fitting uses 2,983 and reserves 745 for calibration; testing uses the same 415 windows. The processed Farm~6 file matches the archived screening records, but has modified entries and irregular timestamps. All 415 test labels agree with the original workbook; this does not prove that all upstream feature processing is causal. These limits are stated explicitly.',
 ['data_partition','experiment_scope']),
'Reviewer 2, Major Comment 5':(
 r'The corrected derivation now treats the source residual mean as a function of the predictor, not a frozen scalar. Data, kernel hyperparameters, centering matrices, and landmarks are fixed within an optimization run. Since $H_S\mathbf1=0$, centering eliminates $b$ exactly and the variance is computed only on source samples. We use $\lambda_1\|H_S(y_S-\Phi_S^Tw)\|^2/(2n_S)$ throughout, define $S=\operatorname{diag}(H_S/n_S,0)$, and retain both its quadratic and label-dependent linear terms. Only $\lambda_1y_S^TH_Sy_S/(2n_S)$ is constant and may be dropped. The domain term is $(w^T(\bar\phi_S-\bar\phi_T))^2=\beta^TKaa^TK\beta$, a prediction-direction mean constraint rather than full feature alignment. The exact kernel QP, fixed Nystr\"{o}m feature approximation, and Adam optimizer are distinguished explicitly. In the explicit formulation both $R$ and $c$ carry $1/n_S$; the code now includes $-c$ and uses batch loss sums plus $(B/N)(Rw-c)$, consistently with the summed-loss objective. Landmark selection uses a seeded source-training subset of at most 2,000 inputs, with a stabilized fixed feature map. The revised code passes small deterministic objective, gradient, batch-scaling, and independent primal--dual checks, including singular Gram matrices. Those algebraic checks are supplemented by a separately recorded Farm~6 fixed-configuration evaluation of the full, no-variance, no-discrepancy, and target-only models. All four share the same 415 test windows, and intervals use 745 reserved target-training labels for calibration. Downstream scaling uses fitting data only, while the supplied workbook has upstream processing and timestamp limitations. Earlier fast-solver scores are not reassigned to this corrected objective, and no Adam convergence rate or approximation-error bound is asserted.',
 ['architecture','objective','variance_roles','domain_roles','optimization_layers','normalization','variance_matrix','mmd_matrix','representer','variance_expansion','kernel_constant','convexity','beta_recovery','dual_matrices','bias_recovery','feature_map','nystrom_R','nystrom_c','nystrom_consistency','pinball_subgradient','batch_gradients','algorithm','convergence','experiment_scope','ablation','framework_0','framework_1','framework_2','framework_3']),
'Reviewer 2, Major Comment 6':(
 r'We have replaced the broad guarantee claims with a direct convexity argument and explicit scope. The assumptions are a positive-semidefinite kernel, $\lambda_1,\lambda_2,C_S,C_T\ge0$, positive total loss weight, and $0<\tau<1$. An RBF kernel has strictly positive bandwidth, with strict positive definiteness requiring distinct inputs. $\Omega$ is positive definite only in the positive-definite-kernel case; a Moore--Penrose inverse handles singular kernels. $R\succeq I$ gives strong convexity in the explicit weight vector for fixed intercept, not joint strong convexity. Landmarks are fixed before optimization, selected from a documented source-training subset, and the feature map uses $10^{-6}$ stabilization. Adam uses a fixed learning rate and finite epochs. A bounded scalar pinball subgradient alone does not bound the full parameter direction, and we no longer assert an $O(1/T)$ rate or a bound between the approximate and exact solutions. The computational discussion also includes dense feature and regularizer costs that the earlier $O(TNm)$ expression omitted.',
 ['objective','convexity','beta_recovery','feature_map','nystrom_R','nystrom_c','batch_gradients','algorithm','complexity_intro','complexity_table','convergence','conclusion_method','conclusion_evidence']),
'Reviewer 2, Minor Comment 1':(
 r'The Abstract has been shortened and now separates the model construction, conditional convexity statement, six-farm screening scope, and empirical evidence limits. It identifies the variance as symmetric and the optimizer as Adam on fixed Nystr\"{o}m features. It no longer asserts verified predictive superiority, a convergence rate, or readiness for deployment of the corrected model.',
 ['abstract','conclusion_method','conclusion_evidence']),
'Reviewer 2, Minor Comment 12':(
 r'The Conclusion now states the single-farm scope of the archived comparisons, the sample-level extreme-event definition, the lack of chronological/event-separated validation, and the absence of deployment-latency evidence. It distinguishes the corrected objective, its implementation checks, and the new fixed-configuration Farm~6 diagnostic from earlier predictive and timing records. The new run shows essentially unchanged point RMSE when the variance penalty is removed and cannot establish chronological generalization.',
 ['conclusion_method','limitations']),
'Reviewer 3, Concern 2':(
 r'Your interpretation is correct: the source residual mean changes with $(w,b)$ and must not be held fixed. Substitution gives $H_S(y_S-\Phi_S^Tw-b\mathbf1)=H_S(y_S-\Phi_S^Tw)$ because $H_S\mathbf1=0$. We now use the normalized centered variance throughout, with $S=\operatorname{diag}(H_S/n_S,0)$. Expanding it retains the cross term involving labels; only the term $\lambda_1y_S^TH_Sy_S/(2n_S)$ is independent of the optimization variables. The explicit-feature counterpart therefore has $c=\lambda_1\Psi_S^TH_Sy_S/n_S$, which is retained in the corrected optimizer. There is no fixed residual-mean initialization or fix-then-freeze step.',
 ['objective','normalization','variance_matrix','variance_expansion','kernel_constant','nystrom_R','nystrom_c','nystrom_consistency','batch_gradients']),
'Reviewer 3, Concern 3':(
 r'We agree that the original fixed-RKHS mean-embedding norm was independent of the prediction coefficients and could not yield the subsequent quadratic penalty. The objective now explicitly uses $(w^T(\bar\phi_S-\bar\phi_T))^2$. With $a=[\mathbf1_{n_S}^T/n_S,-\mathbf1_{n_T}^T/n_T]^T$ and $w=\Phi\beta$, this becomes $(\beta^TKa)^2=\beta^TKM K\beta$, where $M=aa^T$. It is a constraint on the difference between predictive means; it is not full marginal feature alignment and is not asserted to be sufficient for transfer accuracy. The factor $1/2$ is included consistently from the initial objective onward.',
 ['objective','domain_roles','normalization','mmd_matrix','variance_expansion','framework_0']),
'Reviewer 3, Concern 4':(
 r'We accept the correction and now describe the term as symmetric centered residual-variance regularization. It has no explicit $\tau$ dependence, and quantile-dependent asymmetry comes from weighted pinball loss. Its translation invariance permits nonzero residual means, but does not imply preservation of skewness, residual symmetry, Gaussianity, or guaranteed calibration. We have removed the contradictory symmetry-conflict argument throughout the Abstract, Introduction, Related Work, method discussion, and ablation interpretation. The earlier SymVar scores are excluded: the current full residual-variance objective is already symmetric, and the archived component records contain inconsistent point metrics and test-dependent interval post-processing. A new fixed-configuration comparison is now reported for the full, no-variance, no-discrepancy, and target-only variants. The full and no-variance point RMSE values are 7.531988 and 7.531987 MW, respectively, so these data do not demonstrate a benefit from residual-variance regularization at the inherited weight. The full-model calibrated interval has PICP 0.971084 and mean width 27.698317 MW; every interval variant is reported. The evaluation retains the historical split and supplied processed data, and no significant or general component advantage is claimed.',
 ['abstract','intro_challenges','intro_framework','contributions','roadmap','related_variance','variance_roles','ablation','conclusion_method']),
'Reviewer 3, Concern 5':(
 r'The loss-sum formulation means that the aggregate coefficient ratio is $n_TC_T/(n_SC_S)$, so a large $C_T/C_S$ can indeed do more than compensate sample-count imbalance. We have removed the claim that the earlier few-shot comparison established otherwise. The revised method makes this dependence explicit and does not claim an optimal weight ratio. A fixed-test comparison of equal, domain-normalized, imbalance-based, and current weights under the corrected objective remains necessary; the prior few-shot scores cannot answer it because their solver provenance is unresolved. No completed corrected-model weighting sensitivity is claimed.',
 ['weighting','weighting_bullet','fewshot_evidence'])
}
RESPONSES['Reviewer 2, Minor Comment 2']=(
 r'The six-farm superiority wording is not supported. The revised Abstract and contributions now distinguish six-farm screening from the Farm~6 stress-test scope and distinguish archived values from corrected-model validation. We have synchronized those passages and the Conclusion. A separate Highlights source has not been located in the available manuscript files, so we do not claim that an external Highlights document has already been updated.',
 ['abstract','contributions','limitations'])
RESPONSES['Reviewer 1, Concern 2']=(
 r'The adopted threshold definition and the distinction between meteorological extremes and ramp events are stated explicitly. Recalculation from the supplied processed data reproduces all six screening counts and distance statistics. We have also removed the unsupported interpretation of low-power observations as turbine cut-out: Farm~6 has maximum recorded hub wind speed 23.82 m/s and zero samples above the adopted 25 m/s threshold. We agree that changing the thresholds changes the selected target population and can affect both sample size and predictive metrics. No retraining-based threshold sensitivity or event-type breakdown has been completed for the corrected objective; the new fixed-configuration Farm~6 evaluation does not vary those thresholds or separate event types. The Conclusion now explicitly states this limitation, and we do not regard the existing definition alone as empirical validation of threshold robustness.',
 ['threshold_definition','limitations'])
# These related empty entries receive the concrete method/evidence corrections as well.
RESPONSES['Reviewer 1, Concern 9']=(
 r'The earlier training-time table does not establish approximation accuracy or the runtime of the corrected objective. We now account for $O(Nm^2)$ dense transformations and regularizer construction and $O(m^2)$ regularizer multiplication per batch, in addition to kernel evaluation and factorization. We retain the old timings only as archived implementation records and remove the claim that they validate an equal-objective solver comparison. No approximation-error bound is asserted. The requested landmark-size accuracy--runtime experiment remains outstanding and requires new reproducible runs.',
 ['fast_intro','feature_map','complexity_intro','complexity_table','convergence','legacy_scaling_intro','legacy_scaling_discussion'])
# Assign removal evidence to the concern that it directly addresses. Whole old
# experiments are not mechanically repeated under the variance justification.
def response_selections(records):
    selections={k:list(v[1]) for k,v in RESPONSES.items()}
    additions={
        'Reviewer 2, Major Comment 5':['fast_intro','complexity_intro','complexity_table','distribution_shift','experimental_setup','framework_4','primary_implementation_checks'],
        'Reviewer 2, Major Comment 2':['data_sampling','data_quality','training_tail_rules'],
        'Reviewer 1, Concern 2':['training_tail_rules','screened_counts_scope'],
        'Reviewer 1, Concern 3':['screening_interpretation','screening_ranking_scope','original_experiment_1_table_1_removed','original_experiment_1_figure_removed','original_experiment_1_setup_removed','historical_shift_metrics_removed'],
        'Reviewer 2, Major Comment 1':['legacy_point_intro','legacy_point_discussion','legacy_interval_intro','legacy_interval_discussion','interval_nominal_coverage','original_experiment_2_table_1_removed','original_experiment_3_table_1_removed','original_experiment_2_baselines_removed','original_experiment_3_baselines_removed']+['archive_caption_'+str(i) for i in [6,7,8,9,10,11,13]],
        'Reviewer 1, Concern 5':['primary_implementation_checks'],
        'Reviewer 1, Concern 9':['archive_caption_15','original_experiment_5_table_1_removed','original_experiment_5_scales_removed','original_experiment_5_qp_claim_removed'],
        'Reviewer 1, Concern 10':['archive_caption_12','archive_caption_14','original_experiment_2_table_2_removed','original_experiment_3_table_2_removed','original_experiment_2_noise_setup_removed','original_experiment_3_noise_setup_removed'],
    }
    irrelevant_variance_deletions={'legacy_point_intro','legacy_point_discussion','legacy_interval_intro','legacy_interval_discussion','interval_nominal_coverage'}
    selections['Reviewer 1, Concern 5']=[k for k in selections['Reviewer 1, Concern 5']
        if k not in irrelevant_variance_deletions and not k.startswith('archive_caption_')]
    for title,keys in additions.items():selections[title]+=keys
    selections={title:list(dict.fromkeys(keys)) for title,keys in selections.items()}
    used={key for keys in selections.values() for key in keys}
    missing=used-set(records);unused=set(records)-used
    if missing:raise ValueError('Response selections reference missing changes: '+','.join(sorted(missing)))
    if unused:raise ValueError('Changes missing an independent response: '+','.join(sorted(unused)))
    return selections

def default_action(keys,records):
    """Produce a concrete fallback action from the selected change records."""
    kinds={records[key].get('change_type','replacement') for key in keys}
    verbs=[]
    if kinds & {'replacement','relocation'}:verbs.append('revised')
    if 'addition' in kinds:verbs.append('added')
    if 'deletion' in kinds:verbs.append('removed')
    if not verbs:verbs=['updated']
    scopes=[]
    for key in keys:
        location=response_location(records[key])
        # Retain a useful manuscript-level scope while avoiding a long list of
        # repeated paragraph descriptions.
        scope=location.split(', subsection',1)[0].split(', under the subheading',1)[0]
        scope=scope.split(', near',1)[0].split(', in the corresponding',1)[0]
        if scope not in scopes:scopes.append(scope)
    if len(scopes)>4:scope_text=', '.join(scopes[:4])+', and the related passages listed below'
    elif len(scopes)>1:scope_text=', '.join(scopes[:-1])+', and '+scopes[-1]
    else:scope_text=scopes[0] if scopes else 'the relevant manuscript passage'
    return ('We '+', '.join(verbs[:-1])+(' and ' if len(verbs)>1 else '')+verbs[-1]
            +' the relevant text in '+scope_text+'. The exact original wording and revised text are reproduced below at each approximate location.')

def validate_response_format(text,used):
    errors=[]
    if r'Page~\pageref' in text:errors.append('navigation still contains page references')
    if r'\usepackage[hidelinks]{hyperref}' not in text:errors.append('hyperlinks are not configured with hidelinks')
    for key in used:
        begin='% BEGIN VERIFIED CHANGE '+key
        end='% END VERIFIED CHANGE '+key
        if begin not in text or end not in text:
            errors.append('missing rendered change block: '+key);continue
        block=text.split(begin,1)[1].split(end,1)[0]
        if r'\blacktriangleright\blacktriangleright' not in block:
            errors.append('missing approximate location: '+key)
        if r'\textbf{Original text:}' not in block or r'\textbf{Revised text:}' not in block:
            errors.append('missing Original/Revised labels: '+key)
        visible_old=r'\noindent\textbf{'+key.replace('_',' ')+'. Original text:}'
        if visible_old in block:errors.append('internal record name remains visible: '+key)
    if errors:raise ValueError('\n'.join(errors))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--check-records',action='store_true',help='Validate sources, deletion scope, and concern coverage without reading/writing the response or requiring aux files.')
    args=parser.parse_args()
    records=json.loads(RECORDS.read_text())
    updates_path=ROOT/'review/replied_response_updates.json'
    actions={}
    if updates_path.exists():
        for title,update in json.loads(updates_path.read_text()).items():
            if title in RESPONSES:
                previous=RESPONSES[title]
                selected=update.get('changes',previous[1])
                selected=list(dict.fromkeys(selected+update.get('extra_changes',[])))
                RESPONSES[title]=(update.get('response',previous[0]),selected)
            else:
                # New replies can be added in the data file without editing
                # this generator. They must supply both prose and a complete
                # change selection.
                selected=update.get('changes',update.get('extra_changes',[]))
                if not update.get('response') or not selected:
                    raise ValueError('New response requires response and changes: '+title)
                RESPONSES[title]=(update['response'],list(dict.fromkeys(selected)))
            if update.get('action'):actions[title]=update['action']
    validate_records(records)
    selections=response_selections(records)
    if args.check_records:
        print('Verified',len(records),'change records and coverage by',len(selections),'independent responses; repository archives are excluded from the submission.')
        return
    refs={side:load_refs(ROOT/d/'build/TLQLDMR_energy.aux') for side,d in [('original','TLQLDMR_manuscript'),('revised','TLQLDMR_R1')]}
    text=BASE.read_text()
    if r'\usepackage[linesnumbered,ruled]{algorithm2e}' not in text:
        text=text.replace(r'\usepackage{float}',r'\usepackage{float}'+'\n'+r'\usepackage[linesnumbered,ruled]{algorithm2e}')
    if r'\usepackage{adjustbox}' not in text:
        text=text.replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage{adjustbox}')
    # Hyperlinks remain active but use the surrounding black text color.
    text=re.sub(r'\\usepackage(?:\[[^]]*\])?\{hyperref\}',lambda _:r'\usepackage[hidelinks]{hyperref}',text)
    text=text.replace('We have carefully considered all comments and revised the manuscript accordingly. Below we provide a point-by-point response to the comments. In the revised manuscript, all changes are highlighted to facilitate checking.',
        'The methodological revisions and their evidence limits are described below. The revised formulation has been checked on small numerical problems and evaluated with chronological common-setting experiments and a validation-selected six-site comparison. The response reports favorable and unfavorable results, the information used by each protocol, and retained optimization gaps. Changes to the manuscript are marked in blue.')
    # Keep the navigation list but omit page numbers, which become stale as
    # detailed replies are expanded.
    navigation=(r'\textbf{The response letter is structured as follows:}\\'+'\n\n'
        +r'\begin{itemize}'+'\n'
        +r'    \item [$\bullet$]Response for \nameref{edi}.'+'\n'
        +r'    \item [$\bullet$]Response for \nameref{rev1}.'+'\n'
        +r'    \item [$\bullet$]Response for \nameref{rev2}.'+'\n'
        +r'    \item [$\bullet$]Response for \nameref{rev3}.'+'\n'
        +r'\end{itemize}'+'\n\n')
    text=re.sub(r'\\textbf\{The response letter is structured as follows:\}\\\\\s*\\begin\{itemize\}[\s\S]*?\\end\{itemize\}\s*',lambda _:navigation,text,count=1)
    used=set()
    for title,(response,_) in RESPONSES.items():
        if title=='Reviewer 2, General Comments':
            reviewer_start=text.index(r'\label{rev2}')
            h=text.index(r'\subsubsection*{\underline{General Comments}}',reviewer_start)
        else:
            marker=r'\textbf{'+title+'}'
            h=text.index(marker)
        start=text.index(r'\color{black}',h)
        # A few existing actions open an extra group; replace the complete body.
        next_header=text.find(r'\textcolor{customgreen}',start)
        # A rendered change may itself quote a manuscript ``\\section*``
        # (for example, Data and Code Availability).  Using a generic section
        # token as the replacement boundary makes a second generator run leave
        # the tail of the previous body behind.  Reviewer anchors and comment
        # headers are the stable boundaries between response entries.
        next_reviewer=text.find(r'\phantomsection\def\@currentlabelname',start)
        candidates=[v for v in (next_header,next_reviewer) if v>=0]
        stop=min(candidates) if candidates else text.index(r'\bibliographystyle',start)
        pagebreak=text.rfind(r'\newpage',start,stop)
        if pagebreak>=start:stop=pagebreak
        comment=text.rfind('% -------------Reviewer',start,stop)
        if comment>=start:stop=comment
        body='\\color{black}\n\\subsubsection*{\\textbf{\\underline{Author response}}}\n'+response+'\n\n\\color{black}\n\\subsubsection*{\\textbf{\\underline{Author action}}}\n'
        body+=actions.get(title,default_action(selections[title],records))+'\n\n'
        for key in selections[title]:
            used.add(key);record=records[key]
            body+=render_change(key,record,refs)
        text=text[:start]+body+text[stop:]
    text=text.replace(
        'The populated responses document the corrected formulation and the completed chronological study, including its negative results and remaining evidence limits. Unpopulated response entries still require completion before submission.',
        'The responses document the corrected formulation, the completed chronological studies, their unfavorable as well as favorable results, and the remaining evidence limits.')
    unused=set(records)-used
    if unused:raise ValueError('Changes missing an independent response: '+','.join(sorted(unused)))
    validate_response_format(text,used)
    if args.check:
        if text!=BASE.read_text():raise ValueError('Response excerpts or reference numbers need regeneration.')
        print('All',len(records),'Original/Revised blocks match their sources; response entries are current.')
    else:
        BASE.write_text(text)
        print('Generated',len(RESPONSES),'independent responses from',len(records),'verified change blocks.')
if __name__=='__main__':main()
