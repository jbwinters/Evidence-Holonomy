Irreversibility and the Arrow of Time in Physics and Information Theory
Irreversibility – the arrow of time – refers to the asymmetric, one-way direction of processes in time. While the microscopic laws of physics are time-reversible, macroscopic phenomena exhibit a preferred direction associated with entropy increase
journals.aps.org
. In thermodynamics, this arrow of time is formalized by the Second Law, which requires a positive mean entropy production for spontaneous processes
journals.aps.org
. In other words, irreversible behavior usually manifests as the production of entropy in the forward time direction
arxiv.org
. Classic paradoxes (e.g. Loschmidt’s reversibility paradox and Maxwell’s demon) spurred development of statistical mechanics to reconcile microscopic reversibility with macroscopic irreversibility. Boltzmann’s $H$-theorem (1872) was an early quantitative attempt to derive an arrow of time from molecular collisions, introducing a time-monotonic quantity ($H$) analogous to entropy. Modern quantum and classical formulations continue to study how an arrow of time emerges from time-symmetric dynamics
arxiv.org
journals.aps.org
. For example, Fröhlich (2022) discusses how irreversibility arises in quantum systems and notes that irreversible behavior “often manifests itself in the guise of entropy production”
arxiv.org
. Information theory provides complementary insights into time’s arrow. In an information-theoretic sense, time irreversibility can be quantified by how distinguishable a forward-time sequence of events is from its time-reversed counterpart
arxiv.org
. If the statistical properties of a time series remain identical upon time reversal, the process is reversible; any deviation indicates an arrow of time. Equivalently, a time-reversible (equilibrium) process obeys detailed balance (no net probability currents between states), whereas an irreversible process breaks this symmetry
frontiersin.org
. A precise measure of this asymmetry is given by the Kullback–Leibler (KL) divergence between the probability distribution of forward trajectories and that of reverse trajectories
arxiv.org
. Notably, in stochastic thermodynamics it has been shown that the entropy produced by a nonequilibrium process is equal to this KL divergence (often called a “relative entropy” or “entropic distance”) between the forward process and its time-reversed process
journals.aps.org
journals.aps.org
. This deep result, derived in various forms by Crooks, Kawai, Parrondo, and others, establishes a quantitative link between information asymmetry and thermodynamic irreversibility. For instance, Batalhão et al. (2015) experimentally verified that the entropy produced in a microscopic quantum process (a nuclear spin system undergoing a rapid quench) is equal to the KL divergence between the process and its time-reverse
journals.aps.org
. Such findings address the arrow of time “from a microscopic standpoint,” confirming that even at micro scales, entropy production is the arrow of time
journals.aps.org
. In summary, the arrow of time in physics is fundamentally tied to entropy increase and can be understood in statistical–informational terms as the distinguishability of forward vs. backward evolution
arxiv.org
.
Entropy Production: Theory and Estimation Methods
Entropy production (EP) quantifies the degree of irreversibility in thermodynamic processes. For a given system and process, the entropy production $ΔS_{\text{tot}}$ measures how much entropy is generated (in system + environment) beyond what would be conserved in a reversible process. In a stationary Markov process, for example, the steady-state EP rate can be computed from transition probabilities by comparing forward and reverse transition rates: one classic formula is $σ = \tfrac{1}{2}\sum_{i,j} P_i,W(i\to j)\ln\frac{P_i,W(i\to j)}{P_j,W(j\to i)}$, where $W(i\to j)$ is the transition rate from state $i$ to $j$ and $P_i$ the stationary probability of state $i$
arxiv.org
. This formula (from Schnakenberg’s network theory of 1976) makes it explicit that any net imbalance in the probability currents $i\to j$ vs. $j\to i$ contributes to positive entropy production. At equilibrium (detailed balance), the ratio in the log is unity for every pair of states and hence $σ=0$. Positive entropy production is thus a signature of broken time-reversal symmetry at the trajectory level. In practice, directly calculating entropy production requires knowledge of the system’s dynamics (transition rates or equations of motion). However, for many complex or living systems we measure only time-series data without a known model. This has motivated a variety of entropy production inference methods that estimate irreversibility from observed data
nature.com
nature.com
. Several approaches exist:
Model-based calculations: If the dynamical rules are known, one can compute EP from those (as in the Markov formula above or via integrals of $\frac{\delta Q}{T}$ for heat flows). In chemical reaction networks or colloidal particles with known forces, researchers have computed EP by tracking probability fluxes and affinities in phase space
nature.com
. These methods can be data-intensive: estimating stationary distributions and fluxes in a high-dimensional space requires huge sample sizes
nature.com
.
Trajectory perturbation methods: Some techniques derive EP by measuring the system’s response to controlled perturbations. For example, Crooks’ fluctuation theorem and related identities allow one to extract free energy differences (and thus dissipated work) from distributions of forward and reverse work measurements
nature.com
. Similarly, one can perturb a steady state and use linear response theory to estimate EP, though such invasive approaches are not always feasible experimentally
nature.com
.
Model-free time-series methods: A growing body of work aims to estimate entropy production directly from time-series data without assuming a specific model. One strategy is to leverage the variational characterization of EP. For instance, the Thermodynamic Uncertainty Relation (TUR) provides lower bounds on EP in terms of fluctuations of currents; this has been turned into variational inference schemes where one optimizes an observable (or a neural network) to tighten the bound
nature.com
. These schemes project high-dimensional data onto a lower-dimensional current whose statistics are optimized to approach the true EP
nature.com
. Remarkably, such variational methods can give the exact entropy production in the limit of short-time segments for stationary states
nature.com
. Neural-network implementations (e.g. by Otsubo et al., 2022) have demonstrated efficient EP rate estimation in complex systems using this approach
nature.com
nature.com
.
Time-irreversibility metrics: Another model-free approach is to directly quantify the time-reversal asymmetry of recorded trajectories. As discussed, the KL divergence between forward and backward trajectory distributions sets a theoretical benchmark: it equals the total entropy production if we observe all relevant degrees of freedom
arxiv.org
. In partially observed or coarse-grained data, this KL divergence still provides a lower bound on the true entropy production
arxiv.org
. Roldán and Parrondo (2012) introduced techniques to estimate this divergence for finite-state processes, illustrating that the gap between the KL estimate and true EP depends on hidden degrees of freedom
arxiv.org
arxiv.org
. More generally, any statistical measure that vanishes for reversible sequences and is nonzero for irreversible ones can serve as an indicator. Examples include tests based on higher-order moments asymmetry, permutation patterns, or visibility graph differences in time series
pmc.ncbi.nlm.nih.gov
. Porta et al. (2008), for instance, developed multiple statistical tests across embedding dimensions to detect time asymmetry in physiological signals
journals.aps.org
journals.aps.org
. The challenge with direct trajectory-based metrics is bias and sampling error: with limited data, naive estimators of entropy or KL divergence are biased low. To tackle this, researchers have devised bias-correction schemes – e.g. extrapolating estimators as a function of sample size
arxiv.org
 or using surrogate data for significance thresholds
mdpi.com
 (we will discuss concrete applications below).
Recent literature reflects intense activity in developing and comparing such entropy production estimators. Early contributions like Roldán & Parrondo (2010) showed how to estimate dissipation from single stationary trajectories
arxiv.org
. In the last few years, several new methods have appeared: e.g. a model-free local EP measurement in active matter (Ro et al., 2022)
arxiv.org
, a “snippet” approach for time-resolved entropy estimation (van der Meer et al., 2023)
arxiv.org
, and strategies to cope with partial observations and finite data (Fritz et al., 2024; Kapustin et al., 2024)
arxiv.org
. The trend is toward methods that can take raw time-series data (from experiments or simulations) and output an entropy production rate or a statistical score for irreversibility, along with confidence bounds. These tools are essential for applying thermodynamic concepts to complex systems (biological, financial, etc.) where a full model is unavailable. For example, Grandpre et al. (2024) recently described a direct, windowed estimator of time-series irreversibility that corrects for finite-sample bias by extrapolation
arxiv.org
arxiv.org
. Their method reliably detects even subtle arrows of time in data, giving zero (within error bars) for reversible surrogate data and positive values for known nonequilibrium processes
arxiv.org
. Such robust estimation of the “evidence for irreversibility” opens the door to using entropy production as an empirical observable in many fields.
Kullback–Leibler Divergence as a Measure of Time-Reversal Asymmetry
A particularly important asymmetry measure is the Kullback–Leibler divergence (KLD) between forward and reverse trajectory distributions. As noted, KLD provides a quantitative “distance” between the probability of observing a sequence of states in the correct time order versus the reversed order
arxiv.org
. This measure has several desirable properties: it is always non-negative, is zero if and only if the two distributions are identical (perfect time-reversal symmetry), and increases as the time-series exhibits more obvious time-directionality. In the context of stochastic thermodynamics, the KLD between forward and time-reversed paths is equal to the total entropy produced along those paths
journals.aps.org
. Parrondo et al. (2009) succinctly stated that “the entropy production [over a process] and the arrow of time” are directly related, via the distinguishability of the process from its reverse
arxiv.org
. In formula form, for a trajectory $\omega$ over time $[0,t]$, one can show: 
Δ
S
tot
=
k
B
 
D
KL
(
P
[
ω
]
 
∥
 
P
[
ω
~
]
)
,
ΔS 
tot
​
 =k 
B
​
 D 
KL
​
 (P[ω]∥P[ 
ω
~
 ]), where $P[\omega]$ is the path probability and $\tilde{\omega}$ the reversed path, and $k_B$ is Boltzmann’s constant (we will set $k_B=1$ in information units)
journals.aps.org
. This fundamental result was derived by refining fluctuation theorems (e.g. by Kawai, Parrondo, Van den Broeck 2007) and includes the Second Law as a corollary
journals.aps.org
. Batalhão et al. (2015) provided an experimental validation: using an NMR quantum spin system, they measured the nonequilibrium entropy generated after a magnetic field quench and found it equal to the “entropic distance” (relative entropy) between the forward process and its inverse
journals.aps.org
. This affirmed that KLD is not just a formal tool but an experimentally relevant observable for irreversibility. From a data analysis perspective, the KL divergence between forward and backward time series is a model-agnostic metric of time asymmetry. Roldán & Parrondo (2012) demonstrated that for any stationary time series, one can define 
D
KL
(
P
forward
 
∥
 
P
reversed
)
≥
0
,
D 
KL
​
 (P 
forward
​
 ∥P 
reversed
​
 )≥0, and this quantity lower-bounds the physical entropy production of the process generating the series
arxiv.org
. If we have access to all state variables (full information), the inequality becomes an equality
arxiv.org
. In practical terms, if one computes the likelihood of the observed sequence and the likelihood of the time-reversed sequence (under the empirical distribution of sequences), the KL divergence between these tells us how irreversible the series is. A value of zero indicates no preference for forward vs backward (time-symmetric statistics), whereas larger values indicate a stronger arrow of time. Grandpre et al. (2024) refer to this KLD as the “evidence for the arrow of time” and emphasize that it is positive semi-definite and can be estimated from data
arxiv.org
arxiv.org
. They also stress the need for bias corrections: because trajectory space grows exponentially with time length, naive estimators of KLD suffer systematic underestimation. By extrapolating the KLD estimate vs. sample size to infinite data, one can overcome this bias and obtain an accurate irreversibility measure
arxiv.org
. A key point is that KLD-based irreversibility metrics vanish if a system is at equilibrium or obeys detailed balance. For example, when Grandpre et al. tested their estimator on synthetic data from a detailed-balance (reversible) Markov process, the extrapolated KLD converged to zero (within error)
arxiv.org
. This is an important sanity check: a correct irreversibility measure should recognize an equilibrium (no arrow of time) situation. Conversely, for nonequilibrium data, the KLD yields a strictly positive value. In the neural data that Grandpre et al. analyzed (retinal neurons responding to stimuli), they found a small but significant KLD > 0 for the real spike train, indicating a subtle arrow of time in the neural response dynamics
arxiv.org
. They confirmed this irreversibility was not an artifact by showing that a surrogate (randomly time-shuffled) version of the spike train produced KLD ~ 0 as expected
arxiv.org
. In summary, the KL divergence has emerged as a gold-standard measure of time-series irreversibility. It connects directly to entropy production and has been employed both in theoretical derivations and in data-driven studies. One practical implementation is to use machine learning classification: training a classifier to discriminate forwards vs. reversed time series is effectively an empirical way to estimate the KL divergence (since in the limit of a perfect classifier, the log-likelihood ratio recovered is related to the KL divergence). Indeed, recent work has framed irreversibility estimation as a binary classification problem (forward vs reverse) to harness modern ML tools
link.aps.org
. This underscores that KL divergence is central to quantifying the arrow of time – a point echoed by Seifert (2012) in his review of stochastic thermodynamics
arxiv.org
. The equivalence between entropy production and path KL divergence is “central to many recent developments in nonequilibrium statistical physics”
arxiv.org
, providing a unifying language across physics and information theory for discussing irreversibility.
Holonomy and Path-Dependence in Dynamical Systems

https://commons.wikimedia.org/wiki/File:Parallel_Transport.svg
Figure: Illustration of holonomy on a curved surface. If a vector is parallel-transported around a closed loop on a sphere (path A→N→B→A), it ends up rotated by an angle $α$ relative to its initial orientation. This rotation is proportional to the area enclosed by the loop and the sphere’s curvature, reflecting the path-dependence of the process
commons.wikimedia.org
. In general, holonomy measures the net change (e.g. in orientation or phase) gained by transporting an object through a closed path in a space that has curvature or other geometric structure. In dynamical systems, path-dependence means that the outcome of traversing a cycle depends on the path taken, not just the start and end states. A classic example is the Berry phase in quantum mechanics: when a quantum system’s parameters are slowly varied in a cycle, the system can acquire a phase shift (geometric phase) that depends only on the closed path through parameter space, not on the time or speed – this phase is a holonomy in the system’s projective Hilbert space
mdpi.com
. Likewise, in classical mechanics, systems with nonholonomic constraints (e.g. a rolling wheel that cannot slip) have the property that returning to the starting point in configuration space can lead to a different orientation or internal state. (The term “nonholonomic” literally indicates the presence of a non-integrable or path-dependent constraint; a “holonomic” constraint, conversely, can be expressed without path dependence
classe.cornell.edu
.) For instance, driving a car in a loop (a combination of forward/backward and turning motions) can result in a net orientation change – the parallel parking problem is a famous illustration of noncommuting movements. Mathematically, holonomy is closely tied to the concept of curvature. As shown in the figure, transporting a vector around a loop on a curved surface yields a rotation proportional to the surface’s curvature
commons.wikimedia.org
. In differential geometry, one formalizes this with the holonomy group, consisting of all transformations obtained by parallel-transport around closed loops
mathworld.wolfram.com
mathworld.wolfram.com
. If the space is flat (zero curvature), transporting around any contractible loop returns the vector unchanged (zero holonomy). Curvature leads to a nonzero holonomy, which is in fact an alternate way to define curvature (via the Ambrose–Singer theorem relating holonomy and the Riemann curvature tensor)
numberanalytics.com
numberanalytics.com
. In more physical terms: going around the block can leave you changed, if the underlying space or dynamics has some “twist” or nontrivial structure. When we consider observer transformations, holonomy can also arise. An “observer-transported holonomy” framework (as in the user’s context) suggests that if an observer’s frame of reference is carried through a sequence of states or coordinate changes and brought back to the start, the observed system may not return to its original description. This is analogous to gauge fields in physics: for example, if an observer accelerates and then decelerates (taking a loop in velocity-space), certain measured phases (like the Sagnac effect or Thomas precession) indicate a net holonomy. In general, any kind of cyclic operation in a nonlinear system – whether it’s transporting a physical object, rotating a reference frame, or cycling through internal state space – can produce a path-dependent residue. One concrete illustration is the Foucault pendulum: as the Earth rotates (the “observer” reference frame changes), the pendulum’s plane of swing precesses, effectively accruing a holonomy equal to the latitude angle after a full day (closed loop in Earth’s rotation). In summary, holonomy in dynamical systems refers to the idea that history matters: taking one path vs. its time-reverse (or vs. another path between the same endpoints) can lead to different final outcomes. Irreversibility is deeply connected to this concept. An irreversible cycle (such as a thermodynamic cycle with dissipation) is path-dependent – the work done around the loop depends on the direction and sequence, and one cannot recover the initial state functions without an extra cost (like dissipated heat). Indeed, one might view entropy production as a kind of holonomy in thermodynamic state space: if no entropy is produced, the integrals of δQ/T around a cycle sum to zero; if entropy is produced, there is a nonzero “loop integral” – a path-dependent remainder. This analogy is not just poetic; recent theoretical efforts have explored geometric formulations of thermodynamics where dissipative forces are seen as curvatures in a control parameter space, leading to hysteresis loops and irreversible holonomies (e.g. cyclically changing protocol parameters yields a bit of work that cannot be recovered – a loop-dependent loss). The new observer-transported holonomy framework mentioned by the user likely builds on these ideas: by following how an observer’s perspective (or coarse-graining) changes along a system’s evolution, one might define a holonomy that quantifies irreversibility (e.g. using KL divergence as the “length” of the path in probability space). While this is a novel combination of concepts, it resonates with established notions: path dependence, geometric phase, and time’s arrow all intersect in non-equilibrium dynamics.
Applications of Irreversibility and Entropy Production in Physics
In physics, irreversibility and entropy production are core to understanding non-equilibrium phenomena. A vast number of works could be cited; here we highlight a few foundational and recent examples:
Nonequilibrium Statistical Physics: The formalism of stochastic thermodynamics (Seifert 2012, etc.) establishes entropy production as a key quantity for systems ranging from colloidal particles to biomolecular machines
arxiv.org
. Fluctuation theorems (Jarzynski 1997, Crooks 1999) and theorems like Evans–Searles have provided exact relations constraining entropy production fluctuations. These theoretical developments make clear that a positive mean entropy production is equivalent to having an arrow of time; they also provide experimentalists tools to verify thermodynamic principles in small systems. For example, detailed fluctuation theorem experiments with colloidal particles in optical traps have confirmed time-reversal symmetry breaking at the microscopic level by measuring work distributions. Parrondo, Van den Broeck & Kawai (2009) explicitly linked entropy production to the arrow of time by showing that the more distinguishable a forward process is from its reverse, the greater the entropy produced
arxiv.org
.
Microscopic Quantum Experiments: A landmark experiment by Batalhão et al. (2015) used an NMR system (trapped spin-1/2 particles) to measure the arrow of time in a quantum process
journals.aps.org
. They performed a sudden quench (change of magnetic field) and then effectively measured both the forward evolution and the backward evolution (by inverting the sequence of operations). By reconstructing the probability distributions of work in both directions, they could compute the entropy produced. The result – that the mean entropy production equals the KL divergence between forward and backward processes – was one of the first direct demonstrations that the second law can be quantified at the level of path probabilities in a quantum system
journals.aps.org
. This showed experimentally that even an isolated quantum system has a well-defined arrow of time when driven out of equilibrium, consistent with theoretical predictions.
Emergent Irreversibility in Many-Body Systems: In complex interacting systems, one interesting line of research is how local or subsystems’ irreversibility relates to global entropy production. A recent study (Îto & Sagawa, 2020) decomposed entropy production in multipartite systems, showing how an “emergent” arrow of time at a coarse scale can arise even if some micro-details remain hidden. Another 2021 result demonstrated the local emergence of irreversibility: even if a whole closed system is in principle reversible, a local observer (monitoring a subset of degrees of freedom) sees an effectively irreversible behavior with positive entropy production
frontiersin.org
. This ties into the observer perspective: coarse-graining or partial observation tends to induce an arrow of time (since ignored degrees of freedom act as an effective heat bath).
Thermodynamic Engines and Efficiency: In engineering physics, entropy production is directly related to lost work (dissipation) and thus to the efficiency of engines or refrigerators. Modern nano-engines (e.g. single-ion heat engines, quantum dots) operate in regimes where fluctuations are significant, and measuring irreversibility is crucial to evaluate performance. Stochastic thermodynamics has been successfully applied to these systems, with entropy production being estimated from trajectory data of single electron tunneling events or single molecule pulling experiments. The arrow of time here is manifest in violations of detailed balance – e.g. sustained directed currents or cycles (like in molecular motors) indicate ongoing entropy production. The thermodynamic uncertainty relations (TURs) mentioned earlier give one quantitative trade-off: to maintain a certain precision in a current (like rotary motion of F1-ATPase), a minimum entropy production is required
nature.com
. This provides a physics-bound on the performance of microscopic machines.
Active Matter and Living Systems: In systems like bacterial suspensions, self-propelled colloids, or even flocking birds, irreversibility can be observed in the spontaneous breaking of time-reversal symmetry at the collective level. Recent experiments in active matter have measured local entropy production by tracking particle trajectories and identifying probability flux loops (e.g. clusters of particles swirling in a sustained way)
arxiv.org
. Ro et al. (2022), for instance, managed a model-free measurement of local EP in an active granular fluid, finding that regions of high activity correspond to higher time-irreversibility
arxiv.org
. In living cells, non-equilibrium processes abound (metabolism, molecular motors, ion pumps) – detecting the arrow of time in trajectories of, say, mitochondria or vesicles can signal that some driven process is at work. Researchers are beginning to use entropy production as a thermodynamic inference tool in cell biology: if a particular degree of freedom (e.g. a shape oscillation) shows irreversibility, it indicates active driving forces rather than passive equilibrium fluctuations.
In short, physics applications of irreversibility measures span from fundamental tests of quantum thermodynamics to practical diagnostics of engines and complex fluids. Entropy production has become a measurable quantity in experiments, shedding light on how microscopic reversibility gives way to macroscopic irreversibility. As a high-level takeaway: whenever we see a violation of time-reversal symmetry (be it a hysteresis loop in a magnet, a unidirectional current, or decay to equilibrium), we can quantify it by computing entropy produced. The more entropy produced, the more pronounced the arrow of time.
Applications in Neuroscience and Physiology
The concept of irreversibility has found intriguing applications in neuroscience and physiology, where it is used as a marker of complex, nonequilibrium dynamics in living systems. Biological signals are often generated by regulatory feedback loops and nonlinear interactions, which can produce time-asymmetric patterns. By analyzing the time irreversibility of these signals, researchers gain insight into underlying physiological processes and even health states. One application is in heart rate variability (HRV) and other oscillatory physiological signals. Time irreversibility analysis in this context examines whether the statistical properties of, say, beat-to-beat interval sequences remain the same upon time reversal
journals.physiology.org
. If a heartbeat interval series is perfectly time-symmetric, it suggests purely passive, linear fluctuations; any asymmetry indicates nonlinear regulatory inputs (e.g. from the autonomic nervous system). Porta and colleagues introduced measures based on counting upward vs. downward fluctuations and found healthy heart rate dynamics are markedly irreversible, whereas certain pathological conditions or simplified models produce more reversible series
sciencedirect.com
. In short, healthy hearts exhibit time irreversibility due to complex feedback (with parasympathetic and sympathetic interplay), while reduced irreversibility can signal impaired regulation
sciencedirect.com
. Time irreversibility indexes have been proposed as diagnostic metrics for cardiac health, stress, and autonomic balance. In the brain, researchers have applied similar analyses to neural time series such as electroencephalograms (EEG), magnetoencephalograms (MEG), or even spiking activity. A notable study by Zanin et al. (2020) examined resting-state EEG activity in healthy subjects and patients with various neurological disorders
frontiersin.org
frontiersin.org
. They computed time-asymmetry measures (based on permutation patterns of the EEG signal) and found that resting brain activity is generally time-irreversible – indicating that the brain’s spontaneous fluctuations are not simply equilibrium noise but have an inherent directional flow
frontiersin.org
. Intriguingly, they also found that brain pathology was associated with a reduction in time asymmetry
frontiersin.org
. In other words, patients with neurological conditions (they considered Alzheimer’s, epilepsy, etc.) had brain signals that were closer to time-reversible, especially in certain brain regions or frequency bands. This suggests that disease may simplify or linearize the brain’s intrinsic dynamics, making them more thermodynamically “quiet.” These results imply that irreversibility could serve as a biomarker for brain state: for example, one might detect the onset of anesthesia, or distinguish different sleep stages, by observing changes in the irreversibility of EEG patterns (indeed, some studies report higher irreversibility during wakefulness and a drop during deep anesthesia or REM sleep). At the level of single neurons or circuits, irreversibility has also been observed. As mentioned earlier, Grandpre et al. (2024) applied a KL-divergence-based irreversibility measure to spike trains recorded from retinal neurons responding to visual stimuli
arxiv.org
. Even though the neurons’ spiking patterns are noisy, the analysis revealed a consistent arrow of time: the precise timing of spikes carried information that allowed one to statistically distinguish the forward sequence from a time-shuffled one. This asymmetry was attributed to the non-Markovian and history-dependent nature of neural responses (the retina adapts to input, has refractory periods, etc., which all introduce temporal correlations that break time-reversal symmetry)
arxiv.org
. The finding emphasizes that biological computation is inherently nonequilibrium: neurons dissipate energy and produce entropy while processing information, so it is natural that their activity is not time-reversible. Beyond neural signals, many physiological time series have been studied: respiration, blood pressure, gait stride intervals, and more. Time irreversibility is increasingly recognized as a general hallmark of healthy complexity. For instance, Casali et al. (2008) showed that to detect subtle nonlinearities in cardiovascular signals, one often must examine higher-dimensional embedding (considering joint distributions of triplets of points, etc.), because some irreversibility only appears when looking at longer patterns
journals.aps.org
journals.aps.org
. As systems become more complex (with delays, multiple feedback loops), irreversibility may “shift” to higher dimensions, requiring more sophisticated tests
journals.aps.org
. The upshot for physiology is that irreversibility metrics can complement traditional linear measures (like power spectra) to detect meaningful dynamics. In summary, neuroscience and physiology use irreversibility as an index of complexity and active regulation. A high degree of time asymmetry in a biological signal often correlates with a richly coupled, adaptive system (e.g. a brain at wake, a healthy heart under vagal tone), whereas loss of asymmetry can indicate a trend toward disorder or excessive simplification (e.g. neural inactivity in coma or heart failure with reduced variability). Such measures are finding their way into practical biomedical applications, from monitoring depth of anesthesia to early warning signs of seizures or cardiac events.
Applications in Artificial Intelligence and Machine Learning
Irreversibility and the arrow of time have also made inroads into artificial intelligence (AI) research, particularly in areas dealing with sequential data and self-supervised learning. In these contexts, the arrow of time is not a nuisance but a useful signal that algorithms can leverage to learn structure in data. One prominent example is in video analysis and self-supervised representation learning. Videos have a natural arrow of time: certain physical processes in videos only run in one temporal order (for example, a glass shattering or a ball falling obey gravity and thermodynamics, making the forward video look “normal” while the reversed video looks odd). Researchers have exploited this fact by training models to classify whether a video is playing forward or backward, as a way to force the AI to learn meaningful features of dynamics. Wei et al. (2018) introduced an approach called “Learning and Using the Arrow of Time”, in which they compiled large video datasets and trained a convolutional neural network (T-CAM model) to predict the correct time direction of each video clip
ora.ox.ac.uk
. This task doesn’t require manual labels – the videos are trivially labeled by their actual temporal order – so it’s a form of self-supervised learning. The intuition is that to succeed, the model must learn cues of physical causality (e.g. smoke disperses outward over time, objects break into pieces not vice versa, ripples expand on water, etc.). Indeed, the trained model was able to distinguish forward vs. reversed videos with high accuracy, even outperforming human participants on certain rapid or subtle motions
openaccess.thecvf.com
openaccess.thecvf.com
. By doing so, it learned internal representations of motion and cause-effect that proved useful for other tasks (like action recognition) when fine-tuned – thus, the arrow of time served as a useful supervisory signal
ora.ox.ac.uk
ora.ox.ac.uk
. The AI models also gave insights into what visual features indicate time’s arrow. For instance, by visualizing the regions the network attended to, Wei et al. found that things like gravity and dissipation cues were crucial
lilianweng.github.io
. The network learned that water splashing upward then falling, or smoke only spreading (not contracting), or people’s hair and clothing movements in response to momentum, all are tell-tale signs. Low-level physics (gravity, friction) and high-level semantics (an egg un-breaking is unlikely) both play a role
lilianweng.github.io
. This echoes human intuition but the AI can quantify it. Such work even touches on the link between entropy and arrow of time: videos that are time-reversed often contain implausible decreases of entropy (spilled liquids gathering back, etc.), which the network picks up on implicitly. Another use of irreversibility in AI is for generative modeling of realistic video. For a system to generate believable video sequences, it must respect the arrow of time in how events unfold. One recent framework, ArrowGAN (2020), explicitly incorporates a discriminator that learns to classify the arrow of time (forward vs backward) as an auxiliary task
sciencedirect.com
. The generator in the GAN is simultaneously trying to produce videos that not only look real to a static-frame discriminator but also fool the arrow-of-time discriminator. In doing so, the generator is encouraged to produce physically plausible temporal progressions. For example, if the generator started to produce a video of smoke spontaneously clumping together, the arrow-of-time classifier would flag it as “backwards” and the generator would receive a penalty. By contrast, videos that obey normal causal physics are rewarded. This innovation improved the quality of generated videos, demonstrating how embedding physical irreversibility knowledge into training can help AI systems adhere to real-world constraints. Beyond vision, one can consider applications in reinforcement learning and predictive modeling. If an environment or sequence of observations is irreversible, an AI agent might exploit this to infer a direction or to detect novelty. For instance, some anomaly detection algorithms examine time-series for sudden changes in irreversibility – a sudden drop might mean the process became quasi-equilibrium (maybe a machine turned off), whereas a spike might indicate a new driving force appeared. In summary, AI applications treat the arrow of time as both a learning signal and a consistency check. Self-supervised tasks like video direction prediction force neural networks to learn about the world’s dynamics without human labels
ora.ox.ac.uk
. The learned representations have been shown to transfer to improved performance on action recognition and other tasks, confirming that understanding the arrow of time is a part of understanding the physical world. Moreover, generative models that account for time asymmetry produce more realistic outputs
sciencedirect.com
. This is a fascinating cross-pollination: ideas from physics (time’s arrow, entropy) are being used to structure and regularize machine learning models, and conversely, AI is helping to identify which features in data embody the arrow of time.
Applications in Financial Systems and Economics
Financial markets provide an interesting arena for irreversibility analysis. At first glance, an ideal efficient market might be considered time-symmetric (no arbitrage from time direction). However, real financial time series (like stock prices, indices, volatility measures) do exhibit statistical irreversibility, reflecting the presence of nonlinearities, feedback loops, and regime shifts in market dynamics. Researchers in econophysics have applied time-reversal tests to financial data to quantify these effects. One line of work uses permutation and graph-based methods to detect time asymmetry in price series. For example, Zunino et al. (2007) applied permutation entropy and found that financial return series are not invariant under time reversal – essentially, the patterns of ups and downs contain asymmetries beyond what a linear Gaussian model would produce. Similarly, bi-directional tests show that volatility clustering and leverage effects (where volatility reacts differently to price drops vs rises) induce irreversibility in asset returns. In practical terms, stock prices are time-irreversible, and the degree of irreversibility fluctuates over time
sciencedirect.com
. One study explicitly found that years with high irreversibility clustered around periods of market turbulence, while more stable periods showed lower irreversibility
sciencedirect.com
. This already hints at using irreversibility as a diagnostic for market regime changes. A recent comprehensive approach by Fan et al. (2025) introduced a pipeline to detect financial market instabilities via irreversibility analysis
mdpi.com
. They converted financial time series (like stock indexes from 2004–2022) into directed networks using visibility graph algorithms, and then computed the Kullback–Leibler divergence between the network representation of the original series and that of its time-reversed surrogate
mdpi.com
mdpi.com
. Using a sliding window over time, they obtained a time-resolved irreversibility measure (essentially a KLD-based index) for the market. Strikingly, this method successfully identified major market regime shifts and crises, correlating spikes in the irreversibility index with known economic events (e.g. the 2008 financial crisis, flash crashes, COVID-19 crash)
mdpi.com
. Because it is model-free, it picked up anomalies that some econometric models missed. The authors report that the KLD-based irreversibility metric outperformed traditional volatility or moment-based metrics in detecting anomalies, since it captures higher-order temporal structure
mdpi.com
mdpi.com
. In essence, during stable periods the time series looked more reversible (random fluctuations up and down), but when the market was under stress or transitioning, the flow of information and trading caused clear time-asymmetric signatures. They also established a statistical procedure: generate surrogate data that is time-shuffled to break any asymmetry, then see how the real series’ KLD stands out. They chose a threshold (say 90th percentile of surrogate KLD) to flag significant irreversibility
mdpi.com
. A high KLD value means pronounced irreversibility, which signals an abnormal event or transition
mdpi.com
. For example, in 2008 the irreversibility index of the S&P 500 returns shot up well beyond the surrogate-based confidence interval, indicating a clear departure from “business as usual” randomness
mdpi.com
. Such an approach effectively gives an early warning indicator of instability by looking for when the market’s time-series dynamics stop looking like a typical reversible stochastic process. Another intriguing result comes from graph-theoretical analysis: using horizontal visibility graphs, one can map a time series to a network and then check for asymmetries in network motifs forward vs backward. Mitra et al. (2015) found that not only are price series irreversible, but you can cluster periods of high irreversibility which often correspond to market turmoil, while periods of low irreversibility align with calmer markets
sciencedirect.com
. This suggests a qualitative narrative: in crisis times, markets exhibit more directional behaviors (e.g. cascades of selling or rapid recoveries) that imprint an arrow of time on the data, whereas in normal times, fluctuations are more symmetric. From a theoretical standpoint, market irreversibility can be related to the flow of information. Some authors have connected entropy production in economic processes to inefficiencies or agents’ adaptation. If all arbitrage opportunities are removed instantly, one might expect a more reversible (memoryless) price evolution (a pure random walk is reversible in distribution). But when feedback (like trend-following or herding behavior) is present, it introduces memory and cycles that can make the time series irreversible. For instance, a large downward price movement often triggers higher volatility and different recovery dynamics than an upward movement of the same magnitude – this asymmetry (known as the leverage effect) is a source of irreversibility. In summary, financial systems provide both application and analogy for irreversibility concepts. Empirically, irreversibility measures (especially KL divergence and related network entropy measures) have been used as indicators of market regime shifts, crashes, and anomalies
mdpi.com
mdpi.com
. A larger irreversibility index signals more complex, nonequilibrium dynamics – essentially the market is “running hot” with directional structure – whereas a low value suggests more benign, random fluctuations. This is valuable for risk management and analysis of economic systems as complex, driven systems far from equilibrium. It also highlights that even in man-made systems like markets, an arrow of time emerges, measurable by the same tools developed for physical processes.
Irreversibility as a Diagnostic Signal in Complex Systems
As the above examples illustrate, irreversibility measures have broad utility as diagnostics or indicators of underlying changes in complex systems. Because an increase in irreversibility implies a stronger breaking of detailed balance (and typically higher entropy production), it often correlates with the presence of driving forces, feedback, or instability. Conversely, a decrease in irreversibility can indicate a loss of complexity or a transition toward equilibrium-like behavior. This makes time-asymmetry metrics a kind of “thermometer” for a system’s dynamical regime. In financial markets, we saw that spikes in a KL-divergence irreversibility index warned of instabilities or new regimes (e.g. the onset of a crisis)
mdpi.com
mdpi.com
. Because markets are adaptive systems, a regime change (say from bull to bear, or from low to high volatility) involves nonlinear feedbacks that leave an imprint of time asymmetry. An irreversibility analysis can flag these changes earlier or more robustly than linear metrics. This approach can be seen as an information-theoretic early warning signal: when the flow of information in the market (and hence entropy production) surges, the market is not in a steady, time-symmetric state. Practitioners and researchers are interested in such indicators to complement traditional statistical signals. In neuroscience and physiology, irreversibility has been proposed as an indicator of shifts in brain state or health status. For example, a reduction in EEG irreversibility might signal the onset of a pathological state or loss of consciousness
frontiersin.org
. Some studies have suggested using time-asymmetry to monitor patients in intensive care or during anesthesia induction, since it may drop when the brain transitions to an unconscious, more equilibrium-like state. Conversely, a sudden increase in irreversibility in, say, intracranial pressure dynamics could warn of an impending seizure or other critical event, as the system enters a driven, nonequilibrium episode. More generally, healthy physiological function often operates in a balanced yet far-from-equilibrium regime; measuring how far from equilibrium (how irreversible) the dynamics are can provide a single-number summary of system stress or complexity. This is analogous to measuring entropy production in a machine: a well-tuned machine might produce minimal excess entropy, whereas a machine under strain (or about to fail) might produce anomalously high entropy (e.g. through frictional heating, irregular vibrations – which could be detected via time-series irreversibility in sensor data). There is a growing interdisciplinary interest in irreversibility as an early warning signal for critical transitions (sometimes called tipping points) in climate, ecology, and engineering systems. A critical transition often involves the system dynamics reorganizing in a way that might increase entropy production (for instance, the collapse of an ecosystem might be preceded by chaotic, irreversible population fluctuations). While classical early-warning indicators rely on rising variance or autocorrelation, newer methods consider rising entropy production rate as a complementary indicator. If a complex system starts dissipating more energy to maintain its state, it could be a sign of resilience loss. In conclusion, irreversibility metrics like entropy production and KL-divergence-based asymmetry have proven to be insightful diagnostic signals across domains. They condense high-dimensional, nonlinear dynamics into a physically interpretable scalar: how strongly is the arrow of time present? A significant change in this scalar often flags a qualitative change in the system’s behavior. By leveraging high-quality data and advances in estimation techniques, researchers can now monitor this “arrow of time signal” in real time. Whether it is used to anticipate a financial meltdown, to detect a neurological disorder, or to evaluate the stability of an engineered process, the measurement of irreversibility bridges fundamental theory and practical forecasting. As one paper neatly put it, “a larger KLD value indicates that the system exhibits more pronounced irreversibility, making it more likely to detect significant shifts in the time series that signal abnormal events or transitions.”
mdpi.com
 This statement encapsulates why irreversibility is such a powerful concept: it doesn’t just tell us about thermodynamics – it tells us when a complex system is deviating from the ordinary and potentially entering a new regime. References: The content above synthesizes foundational works and recent research from statistical physics, stochastic thermodynamics, information theory, and various applied domains. Key references include theoretical expositions of entropy production and time’s arrow
arxiv.org
arxiv.org
, experimental confirmations of the KLD–entropy production relationship in quantum systems
journals.aps.org
, methods for estimating entropy production from time-series data
nature.com
arxiv.org
, and studies using irreversibility measures in neuroscience
frontiersin.org
, AI
ora.ox.ac.uk
, and finance
mdpi.com
mdpi.com
, among many others as cited inline. These works collectively paint a rich picture of how the arrow of time can be quantified and employed as both a theoretical tool and a practical signal across scientific disciplines.
Citations

Irreversibility and the Arrow of Time in a Quenched Quantum System | Phys. Rev. Lett.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.190601

[2202.04619] Irreversibility and the Arrow of Time

https://arxiv.org/abs/2202.04619

Irreversibility and the Arrow of Time in a Quenched Quantum System | Phys. Rev. Lett.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.190601

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Frontiers | Time Irreversibility of Resting-State Activity in the Healthy Brain and Pathology

https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.01619/full

Irreversibility and the Arrow of Time in a Quenched Quantum System | Phys. Rev. Lett.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.190601

Dissipation: The Phase-Space Perspective | Phys. Rev. Lett.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.080602

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

[1201.5613] Entropy production and Kullback-Leibler divergence between stationary trajectories of discrete systems

https://arxiv.org/abs/1201.5613

[1201.5613] Entropy production and Kullback-Leibler divergence between stationary trajectories of discrete systems

https://arxiv.org/abs/1201.5613
Assessing Time Series Reversibility through Permutation Patterns

https://pmc.ncbi.nlm.nih.gov/articles/PMC7513188/

Multiple testing strategy for the detection of temporal irreversibility in stationary time series | Phys. Rev. E

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.77.066204

Multiple testing strategy for the detection of temporal irreversibility in stationary time series | Phys. Rev. E

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.77.066204

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Dissipation: The Phase-Space Perspective | Phys. Rev. Lett.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.080602

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Functional decomposition and estimation of irreversibility in time ...

https://link.aps.org/doi/10.1103/PhysRevE.110.064310

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

Direct estimates of irreversibility from time series

https://arxiv.org/html/2412.19772v1

File:Parallel Transport.svg - Wikimedia Commons

https://commons.wikimedia.org/wiki/File:Parallel_Transport.svg

Modeling Time’s Arrow

https://www.mdpi.com/1099-4300/14/4/614
[PDF] Notes on non-holonomic constraints - CLASSE (Cornell)

https://www.classe.cornell.edu/~pt267/files/teaching/P3318S13/Sec_05_nonholonomic.pdf

Holonomy Group -- from Wolfram MathWorld

https://mathworld.wolfram.com/HolonomyGroup.html

Holonomy Group -- from Wolfram MathWorld

https://mathworld.wolfram.com/HolonomyGroup.html

Mastering Holonomy: A Deep Dive

https://www.numberanalytics.com/blog/mastering-holonomy-deep-dive

Mastering Holonomy: A Deep Dive

https://www.numberanalytics.com/blog/mastering-holonomy-deep-dive
Temporal asymmetries of short-term heart period variability are ...

https://journals.physiology.org/doi/pdf/10.1152/ajpregu.00129.2008

Heart rate time irreversibility is impaired in adolescent major ...

https://www.sciencedirect.com/science/article/abs/pii/S0278584612001583

Frontiers | Time Irreversibility of Resting-State Activity in the Healthy Brain and Pathology

https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.01619/full

Frontiers | Time Irreversibility of Resting-State Activity in the Healthy Brain and Pathology

https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.01619/full

Multiple testing strategy for the detection of temporal irreversibility in stationary time series | Phys. Rev. E

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.77.066204

Multiple testing strategy for the detection of temporal irreversibility in stationary time series | Phys. Rev. E

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.77.066204
Learning and using the arrow of time - ORA - Oxford University Research Archive

https://ora.ox.ac.uk/objects/uuid:53e35755-a037-4c1e-9f16-8bacc1f9923d

Learning and Using the Arrow of Time

https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Learning_and_Using_CVPR_2018_paper.pdf

Learning and Using the Arrow of Time

https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Learning_and_Using_CVPR_2018_paper.pdf
Learning and using the arrow of time - ORA - Oxford University Research Archive

https://ora.ox.ac.uk/objects/uuid:53e35755-a037-4c1e-9f16-8bacc1f9923d

Self-Supervised Representation Learning - Lil'Log

https://lilianweng.github.io/posts/2019-11-10-self-supervised/

ArrowGAN : Learning to generate videos by learning Arrow of Time

https://www.sciencedirect.com/science/article/abs/pii/S0925231221000643

Irreversibility of financial time series: A graph-theoretical approach

https://www.sciencedirect.com/science/article/pii/S0375960116002401

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Instability of Financial Time Series Revealed by Irreversibility Analysis

https://www.mdpi.com/1099-4300/27/4/402

Irreversibility of financial time series: A graph-theoretical approach

https://www.sciencedirect.com/science/article/pii/S0375960116002401

Estimating time-dependent entropy production from non-equilibrium trajectories | Communications Physics

https://www.nature.com/articles/s42005-021-00787-x?error=cookies_not_supported&code=480fabb9-af73-498e-a98b-cabf58d7ae91

