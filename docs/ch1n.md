### Introduction: The Accelerating Trajectory of AI and the Quest for AGI/ASI

#### Overview of AI’s Progress

Artificial Intelligence (AI) has transitioned decisively from the
speculative realms of science fiction into a tangible, dynamic force
reshaping the contemporary world. Its influence is no longer subtle;
sophisticated AI systems are increasingly integrated into critical
infrastructure and daily life. Autonomous vehicles demonstrate complex
navigational capabilities on public roads, while advanced language
models engage in nuanced, context-aware conversations that were
unimaginable just a few years prior. These examples are but surface
manifestations of a deeper, accelerating trend: the undeniable and
rapidly advancing capabilities of AI systems across a diverse array of
domains. This acceleration signifies not merely incremental improvements
but potentially fundamental shifts in technological paradigms. The sheer
pace of development underscores the urgency and importance of
understanding the trajectory of AI, its underlying principles, and its
ultimate potential.

#### Introducing AGI and ASI

The current wave of AI advancements inevitably leads to considerations
of more profound forms of machine intelligence: Artificial General
Intelligence (AGI) and Artificial Superintelligence (ASI). AGI
represents a significant milestone – the development of AI systems
possessing cognitive capabilities comparable to humans across a broad
spectrum of intellectual tasks, rather than being confined to narrow
specializations.1 It signifies an ability to learn, reason, adapt, and
apply knowledge flexibly in novel situations, mirroring human
versatility. Beyond AGI lies the concept of ASI, defined as an intellect
that dramatically surpasses the cognitive performance of humans in
virtually all domains of interest.1 The pursuit of AGI and ASI
transcends conventional engineering; it represents a fundamental
scientific quest to understand and replicate the essence of intelligence
itself. The potential consequences of achieving AGI, and particularly
ASI, are immense, carrying the possibility of unprecedented societal
transformation, both beneficial and potentially disruptive
(([Schneppat](#ref-schneppatTopicsASI))).

### Setting the Stage: Navigating the Path Forward

This report aims to chart a course through the complex landscape of AI
development, tracing a potential roadmap from the current
state-of-the-art towards AGI and ASI. It will delve into the critical
concepts, historical lessons, contemporary methodologies, and future
enabling technologies that define this journey. A central theme is the
"Bitter Lesson" (([Sutton 2019](#ref-sutton2019bitter))), an influential
insight articulated by AI pioneer Richard S. Sutton, which offers a
compelling, albeit sometimes counterintuitive, perspective on the most
effective strategies for building truly intelligent systems.
Understanding this lesson, alongside the intricacies of modern Large
Language Models (LLMs), the evolution of reasoning capabilities, the
potential of recursive self-improvement, and the role of synthetic data,
is crucial for navigating the path ahead. The scale and acceleration of
current AI progress lend credence to the idea that AGI/ASI are not
merely distant theoretical possibilities but potentially achievable
horizons, demanding rigorous analysis and foresight. However, this
journey is marked by unexpected discoveries and profound uncertainties.
Framing this exploration requires acknowledging both the remarkable
advancements and the inherent complexities and potential risks involved,
fostering a nuanced perspective that balances ambition with caution.

## The Bitter Lesson Revisited: Computation, Search, and Learning as Foundational Pillars

### The Core Argument of the Bitter Lesson

Decades of AI research have yielded numerous approaches, but a recurring
pattern, articulated by Richard S. Sutton as "The Bitter Lesson"
(([Sutton 2019](#ref-sutton2019bitter))), suggests a fundamental
principle governing long-term success. The core argument is starkly
simple yet profound: "general methods that leverage computation are
ultimately the most effective, and by a large margin" . This assertion
is grounded in the observation that computational power has historically
increased exponentially, a trend often associated with Moore’s Law but
extending beyond it to the general cost-effectiveness of computation.
Sutton posits that AI methods designed to harness this ever-expanding
computational resource inevitably outperform approaches that rely
heavily on incorporating human-derived knowledge or crafting
domain-specific heuristics.

Historically, many AI researchers intuitively sought to build
intelligence by encoding their own understanding of a problem domain
into the system. This "human-knowledge-based" approach involved
designing algorithms based on how humans think they solve problems,
embedding expert knowledge for tasks ranging from game playing to
language understanding. This methodology felt natural and often yielded
satisfying short-term progress. However, the Bitter Lesson reveals that
such approaches consistently hit performance ceilings; the very human
knowledge built in eventually becomes a limitation. Breakthroughs,
conversely, have consistently arisen from general-purpose methods that
scale effectively with computation.

The "bitterness" of this lesson stems from its challenge to
human-centric perspectives on intelligence. It implies that the most
powerful AI systems might not operate based on principles easily
understandable or analogous to human cognition. Success often arrived
via methods initially dismissed or viewed with dismay by researchers
invested in replicating human thought processes, leading to a success
"tinged with bitterness" because it triumphed over favored,
human-centric approaches. This recurring historical pattern suggests
that anthropomorphic biases can hinder progress and that embracing
computationally-driven, general methods is key, even if they seem less
intuitive or elegant from a human perspective.

### Historical Case Studies

The validity of the Bitter Lesson is vividly illustrated across several
pivotal domains in AI history. These examples demonstrate a consistent
pattern where initial human-knowledge-based approaches were eventually
superseded by methods leveraging large-scale computation through search
and learning.

#### Computer Chess (Deep Blue vs. Kasparov)(([Pandolfini 1997](#ref-pandolfini1997kasparov)))

The quest to conquer chess was initially dominated by attempts to codify
human chess expertise – strategic principles, tactical motifs, and
expert evaluations. However, the watershed moment arrived in 1997 when
IBM’s Deep Blue defeated the reigning World Chess Champion, Garry
Kasparov, in a six-game match under tournament conditions. Deep Blue’s
victory was not achieved by mimicking human intuition but by leveraging
massive computational power for deep search. The system could evaluate
an astonishing 200 million chess positions per second, utilizing
parallel processing across 32 processors, translating to a speed of
11.38 gigaflops. Many in the computer chess community, invested in
knowledge-based methods, initially disparaged Deep Blue’s approach as
mere "brute force", arguing it lacked true chess understanding and
wasn’t how humans played. Despite this resistance, the outcome was
undeniable. Deep Blue’s win, the first defeat of a reigning world
champion by a computer in a standard match 7, signaled the ascendancy of
computation-heavy strategies and marked a significant validation of the
Bitter Lesson’s principles.

#### Computer Go (AlphaGo/AlphaGo Zero)(([Silver et al. 2017](#ref-silver2017mastering); [Dalgaard et al. 2020](#ref-dalgaard2020global)))

The ancient game of Go, with its vastly larger search space
($`10^{170}`$ possible positions compared to chess’s estimated
$`10^{120}`$), presented an even more formidable challenge. Initial
efforts, mirroring the early chess approach, focused on encoding human
Go strategies and intuition, but these systems struggled against even
amateur players. The breakthrough came with DeepMind’s AlphaGo. It
combined deep neural networks (a policy network to suggest moves and a
value network to evaluate positions) with a sophisticated Monte Carlo
Tree Search (MCTS) algorithm. Crucially, AlphaGo was trained not only on
human expert games (supervised learning) but also through reinforcement
learning by playing millions of games against itself (self-play). This
potent combination of search and learning, fueled by significant
computational resources, enabled AlphaGo to defeat world Go champions
like Lee Sedol. The subsequent development, AlphaGo Zero, took this a
step further, learning entirely from self-play reinforcement learning,
without any human game data. Starting tabula rasa, it surpassed the
original AlphaGo, winning 100-0 in internal matches. AlphaGo Zero’s
success powerfully demonstrated that general learning and search
methods, scaled with computation, could discover superhuman strategies
in highly complex domains, even without relying on human expertise,
further reinforcing the Bitter Lesson.

#### Speech Recognition (ASR) (([Yu and Deng 2016](#ref-yu2016automatic)))

A parallel narrative unfolded in Automatic Speech Recognition. Early
systems relied heavily on human knowledge of linguistics, phonetics, and
the mechanics of the human vocal tract. These rule-based systems, and
early attempts like Bell Labs’ "Audrey" (1952) recognizing digits , were
gradually overtaken by statistical methods, most notably Hidden Markov
Models (HMMs), which gained prominence in the 1970s and 1980s.3 HMMs
leveraged increased computational power and statistical learning from
data, modeling speech as probabilistic sequences rather than relying
solely on predefined rules. The introduction of n-gram language models
further enhanced these statistical approaches. The most recent and
dramatic shift came with the rise of deep learning, particularly using
architectures like Long Short-Term Memory (LSTM) networks and later
Transformers. These deep neural networks, trained on massive datasets
using vast computational resources, achieved unprecedented accuracy.
They rely far less on explicit, hand-engineered human knowledge about
language structure, instead learning intricate patterns directly from
data through general learning algorithms like gradient descent and
backpropagation. This evolution from rule-based systems to HMMs to deep
learning perfectly illustrates the Bitter Lesson’s trajectory: a
progressive shift away from human-encoded knowledge towards general,
computationally intensive learning methods.

<figure>
  <img src="https://arxiv.org/html/2410.09649v1/x6.png" alt="Image description">
  <figcaption>Figure 1:Line plot showing the average alignment scores across years for CVPR papers from 2005 to 2024 (Yousefi and Collins 2024).
</figcaption>
</figure>

Fig. 1 illustrates the evolving alignment of Computer Vision and Pattern
Recognition (CVPR) research with five key "bitter lesson" dimensions
over this period (([Yousefi and Collins
2024](#ref-yousefi2024learning))). These alignment scores are calculated
by averaging ratings from three large language models, which assigned a
0-10 Likert score to sampled paper titles and abstracts for each of the
five dimensions (Learning Over Engineering, Search Over Heuristics,
Scalability With Computation, Generality Over Specificity, and Favoring
Fundamental Principles). The line plot then displays the yearly average
of these dimension-specific scores, aggregated across all sampled papers
and all three LLMs for each year. Vertical lines marking influential
machine learning paper publications provide context to the trends
observed. Notably, the figure reveals consistent upward trends for the
"Scalability with Computation" and "Learning Over Engineering"
dimensions

### The Two Pillars: Search and Learning

The Bitter Lesson consistently points towards two fundamental classes of
general-purpose methods that are uniquely positioned to benefit from and
scale with increasing computational power: search and learning.

#### Search

In the context of AI, search involves the systematic exploration of a
vast space of possibilities—be it potential game moves, reasoning steps,
or solutions to a problem—to identify an optimal or satisfactory
outcome. Deep Blue’s chess prowess was a direct result of its ability to
search through millions of potential move sequences. AlphaGo employed
sophisticated Monte Carlo Tree Search to navigate the enormous state
space of Go.The effectiveness of search algorithms is intrinsically tied
to computational resources; more compute allows for deeper exploration
(looking further ahead) and broader exploration (considering more
alternatives at each step), leading to the discovery of better
solutions.

#### Learning

Machine learning, especially deep learning, enables AI systems to
acquire knowledge and improve their performance autonomously by
processing data and experience, rather than relying solely on explicit
programming. Algorithms learn to identify complex patterns,
correlations, and representations within massive datasets. AlphaGo
learned Go evaluation functions through self-play, and modern ASR
systems learn the nuances of speech from vast audio corpora. Increased
computation directly translates to the ability to train larger, more
complex models on larger datasets for longer periods, enabling the
learning of more sophisticated and accurate world models or policies.

Crucially, both search and learning are general methods. They are not
inherently limited to specific domains or reliant on pre-existing human
knowledge about those domains. They function as meta-methods—powerful
techniques capable of discovering and capturing the inherent, often
"irredeemably complex," structure of the world, rather than being
constrained by simplified human conceptualizations of that structure.
Their ability to scale with computation makes them the primary engines
driving progress towards more capable AI.

### Enduring Relevance and the "Era of Experience" (([Silver and Sutton 2025](#ref-silver2025welcome)))

The insights encapsulated in the Bitter Lesson are not mere historical
footnotes; they remain profoundly relevant in the current AI landscape.
The recent dramatic advances in Large Language Models (LLMs) and other
foundation models are, in many ways, a testament to the lesson’s
principles. The success of models like GPT-4, Llama, and Claude stems
directly from leveraging unprecedented computational scale applied to
general methods—specifically, the Transformer architecture combined with
massive unsupervised learning (pre-training) on internet-scale datasets.
These models rely less on explicitly encoded linguistic rules or
knowledge structures and more on what can be learned through computation
applied to data.

Ongoing research continues to validate and explore the implications of
the Bitter Lesson. Studies analyze the historical alignment of research
trends (e.g., in computer vision) with Sutton’s principles, finding a
clear shift towards general methods and deep learning over time
(([Yousefi and Collins 2024](#ref-yousefi2024learning))). Furthermore,
discussions about potential future "bitter lessons" suggest that the
core tension between human-centric design and scalable computation
remains active.

<figure>
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HFSnTAhd-EvhYXCsS-gMbA.png
  " alt="Image description">
  <figcaption>Figure 2: A sketch chronology of dominant AI paradigms. The y-axis suggests the proportion of the field’s total effort and computation that is focused on RL (David Silver and Richard S. Sutton, 2025). </figcaption>
</figure>


Building upon these principles, Sutton, along with David Silver (a key
figure behind AlphaGo), proposed that we are entering the "Era of
Experience". This perspective further refines the Bitter Lesson’s
emphasis on learning, highlighting the critical role of learning
directly from interaction with an environment (i.e., experience) rather
than solely from static, pre-collected datasets. Reinforcement Learning,
where agents learn through trial-and-error by receiving feedback from
their actions, embodies this principle. The "Era of Experience" suggests
that the most capable AI systems will increasingly be those that can
generate their own data through interaction, leveraging computation not
just to process vast datasets but to actively explore and learn from the
consequences of their actions in complex environments (real or
simulated). This aligns perfectly with the success of systems like
AlphaGo Zero, which learned entirely from self-play experience, and
points towards the importance of RL and agent-based learning in the
pursuit of AGI. As AI development progresses towards AGI, the principle
of prioritizing methods that effectively harness computation through
search and learning—increasingly emphasizing learning from direct
experience—is likely to remain a crucial guiding star. The lesson
teaches a form of humility: acknowledging the limitations of human
intuition in designing complex intelligent systems and recognizing the
immense power unlocked by letting computation-driven general methods
discover solutions through data and interaction.

## Architecting Intelligence: Training and Aligning Large Language Models

The development of modern Large Language Models (LLMs) provides a
compelling contemporary case study in applying the principles of
leveraging computation and general learning methods. These models,
forming the backbone of many current AI applications, undergo a complex
multi-stage training process that begins with a massive pre-training
phase and is followed by careful alignment procedures to shape their
behavior.

### The Foundation: Pre-training Large Models

Pre-training constitutes the initial, computationally demanding phase
where LLMs acquire their foundational understanding of language and the
world. Models like GPT-4 are trained on vast corpora of text and code,
often comprising trillions of tokens sourced from the internet and
licensed datasets. The core objective during pre-training is deceptively
simple: next token prediction. Given a sequence of text, the model
learns to predict the statistically most likely next word or sub-word
unit (token).

<div style="display: inline-flex; align-items: center;">
  <!-- Video Thumbnail -->
  <a href="https://www.youtube.com/watch?v=kYWUEV_e2ss" target="_blank" style="display: inline-block;">
    <img src="https://img.youtube.com/vi/kYWUEV_e2ss/0.jpg" style="width: 100%; display: block;">
  </a>

  <!-- Play Button -->
  <a href="https://www.youtube.com/watch?v=kYWUEV_e2ss" target="_blank" style="display: inline-block;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_play_button_icon_%282013%E2%80%932017%29.svg" 
         style="width: 50px; height: auto; margin-left: 5px;">
  </a>
</div>

This simple objective, however, acts as a powerful incentive structure,
compelling the model to learn a wide range of underlying capabilities to
minimize its prediction error across diverse contexts. As articulated by
researchers like Hyung Won Chung (([Chung
2025](#ref-Chung2025youtube))), to accurately predict the next token in
trillions of different scenarios, the model is implicitly forced to
learn grammar, syntax, semantics, factual world knowledge, and even
rudimentary reasoning abilities. This "inspiration model" approach
contrasts with explicitly "teaching" the model specific skills. Instead,
general abilities emerge spontaneously as necessary tools to succeed at
the broad, computationally intensive task of next-token prediction. The
advantages of this incentive-driven approach include superior
scalability (larger models and datasets lead to better performance),
generality (learned skills are broadly applicable), and the potential
for emergence—the appearance of unexpected capabilities at scale.

The effectiveness of pre-training is strongly governed by Scaling Laws
(([Kaplan et al. 2020](#ref-kaplan2020scaling))). Empirical research has
consistently shown that LLM performance, often measured by the
pre-training loss (prediction error), improves predictably as a
power-law function of three key factors: the size of the model (number
of parameters), the size of the training dataset, and the amount of
computational resources invested in training. These laws provide a
degree of predictability, allowing researchers to estimate the
performance of larger models based on smaller-scale experiments and
guide resource allocation. However, scaling also presents significant
challenges, including immense computational cost, the need for vast
amounts of high-quality data, the potential for learned biases present
in the data, and the environmental impact of large-scale training runs.

A fascinating consequence of scaling is the phenomenon of Emergent
Abilities (([Schaeffer, Miranda, and Koyejo
2023](#ref-schaeffer2023emergent))). These are capabilities, such as
complex arithmetic, multi-step reasoning, or sophisticated in-context
learning, that appear somewhat abruptly in larger models but are absent
or perform at random-chance levels in smaller models. While initially
defined based on model scale, recent work suggests these abilities might
be better understood as emerging when a model’s pre-training loss falls
below a certain task-specific threshold. There is ongoing debate about
whether these abilities are truly emergent properties of scale or
potentially artifacts of how performance is measured or the
non-linearity of certain tasks. Regardless of the precise mechanism, the
existence of emergent abilities highlights the transformative, and
sometimes unpredictable, effects of scaling LLMs, complicating both
capability forecasting and safety assessments.

### Shaping Behavior: The Multi-Stage Alignment Pipeline (Tülu 3 Example (([Lambert et al. 2024](#ref-lambert2024t))))

While pre-training endows LLMs with broad knowledge and linguistic
competence, the resulting models are often unaligned with specific human
goals or preferences. They might generate factually incorrect,
unhelpful, unsafe, or poorly formatted text. To address this,
pre-trained models undergo post-training, a crucial multi-stage process
designed to refine their behavior and instill desired characteristics
like helpfulness, honesty, and harmlessness. The Tülu 3 project from the
Allen Institute for AI provides a valuable, fully open-source example of
a modern post-training pipeline, demonstrating the techniques used to
transform base models (like Llama 3.1) into state-of-the-art
instruction-following assistants.

<figure>
  <img src="https://arxiv.org/html/2411.15124v5/x9.png
  " alt="Image description">
  <figcaption>Figure 3: An overview of the Tülu 3 recipe. This includes: data curation targeting general and target capabilities, training strategies and a standardized evaluation suite for development and final evaluation stage. (Lambert et al. 2024). </figcaption>
</figure>


The Tülu 3 pipeline, representative of many contemporary approaches,
involves several key stages:

1.  **Data Curation
    (([**baack2025bestpracticesopendatasets?**](#ref-baack2025bestpracticesopendatasets)))**
    This foundational stage emphasizes the critical importance of
    high-quality data. It involves sourcing diverse prompts from public
    datasets, carefully analyzing their provenance and licenses, and
    performing aggressive decontamination to remove any overlap with
    downstream evaluation benchmarks, ensuring fair assessment. A
    significant component is the synthetic generation of prompts
    specifically targeting core skills identified as important (e.g.,
    reasoning, math, coding, safety, precise instruction following).
    This targeted data allows for focused improvement in desired
    capabilities.

2.  **Supervised Fine-Tuning (SFT) (([Dong et al.
    2024](#ref-dong2024abilitieslargelanguagemodels)))** In this stage,
    the pre-trained model is fine-tuned on a dataset of high-quality
    prompt-completion pairs. The goal is to teach the model the desired
    format for responding to instructions and to instill basic
    instruction-following capabilities. This involves carefully
    selecting and mixing different SFT datasets (e.g., combining general
    knowledge with coding or math examples) and meticulous
    hyperparameter tuning to optimize performance on target skills
    without degrading others. Evaluation on a dedicated suite guides
    this iterative process.

3.  **Preference Tuning (RLHF/DPO)** (([Korbak et al.
    2023](#ref-korbak2023pretraining))) SFT alone may not capture the
    nuances of human preferences regarding helpfulness, harmlessness, or
    stylistic qualities. Preference tuning aims to align the model more
    closely with these subjective judgments. This typically involves
    collecting preference data, where humans (or increasingly, powerful
    LLMs acting as judges) rank different model responses to the same
    prompt. While Reinforcement Learning from Human Feedback (RLHF),
    often using algorithms like Proximal Policy Optimization (PPO
    (([Schulman et al. 2017](#ref-schulman2017proximal)))) to train the
    LLM against a learned reward model predicting preferences, was
    historically common, many pipelines, including Tülu 3, have also
    employed Direct Preference Optimization (DPO (([Rafailov et al.
    2023](#ref-rafailov2023direct)))). DPO offers a more stable and
    computationally efficient way to directly optimize the language
    model policy to satisfy the observed preferences, typically using a
    binary cross-entropy loss formulation on preferred/dispreferred
    response pairs. Tülu 3 utilizes both off-policy (from other models)
    and on-policy (from Tülu models) preference data, including
    synthetically generated data rated by an LLM judge. Careful data
    mixing and experimentation are crucial for effective preference
    alignment.

4.  **Advanced Techniques (e.g., RLVR (([Mroueh
    2025](#ref-mroueh2025reinforcement))))** Some pipelines incorporate
    additional stages to further enhance specific, often objectively
    verifiable, skills. Tülu 3 introduces Reinforcement Learning with
    Verifiable Rewards (RLVR). Instead of relying on a learned reward
    model approximating human preference, RLVR focuses on tasks where
    correctness can be definitively verified (e.g., mathematical
    problem-solving, executing specific instructions). The model
    receives a positive reward only if its output is verified as correct
    according to ground truth. This provides a strong, targeted signal
    for improving performance on these specific benchmarks, and Tülu 3
    found it significantly boosted scores on MATH, GSM8K, and IFEval,
    even when applied after DPO.

5.  **Evaluation
    (([**cao2025generalizableevaluationllmera?**](#ref-cao2025generalizableevaluationllmera)))**
    Throughout the entire pipeline, a rigorous and standardized
    evaluation framework (like Tülu 3 Eval) is essential. This involves
    using a diverse suite of benchmarks covering the targeted core
    skills, carefully separating development sets from unseen test sets
    to measure generalization, and ensuring evaluation protocols are
    fair and reproducible. Continuous evaluation guides data curation,
    model selection, and hyperparameter tuning at each stage.

This multi-stage process highlights that creating capable and aligned
LLMs is far more complex than simple pre-training. It involves a
sophisticated interplay of supervised learning, preference optimization,
and sometimes targeted reinforcement learning, all heavily reliant on
meticulous data curation, synthesis, and ongoing evaluation. The trend
towards open recipes like Tülu 3 is crucial for enabling broader
research and understanding of these critical post-training techniques.

The architecture of modern AI, particularly LLMs, demonstrates a
powerful synergy between the principles of the Bitter Lesson and
sophisticated engineering. The reliance on massive computation for
pre-training via the general method of next-token prediction allows for
the emergence of complex capabilities without extensive hand-coding of
knowledge. The subsequent alignment pipeline then carefully shapes these
capabilities using a combination of supervised signals, preference
feedback, and targeted reinforcement, layering constraints to produce
models that are not only knowledgeable but also helpful, harmless, and
aligned with human intent. This intricate process underscores that
progress relies both on embracing scalable, general learning principles
and on developing nuanced techniques for guidance and control.
Furthermore, the increasing emphasis on data curation, synthesis, and
decontamination suggests that managing the information diet of these
models is becoming as critical as the learning algorithms themselves,
potentially representing the next major frontier—or bottleneck—in AI
development, especially as readily available high-quality web data
becomes scarcer.

## Beyond Static Knowledge: The Rise of Learned Reasoning

While pre-training equips LLMs with vast amounts of knowledge
assimilated from data, and alignment fine-tunes their behavior,
achieving true AGI necessitates robust reasoning capabilities—the
ability to perform multi-step inference, solve complex problems, and
adapt strategies dynamically. Historically, AI relied heavily on
explicit search algorithms for problem-solving, but these methods face
significant limitations in scaling to the complexity and ambiguity of
real-world reasoning tasks. This has spurred a shift towards
learning-based approaches, particularly using Reinforcement Learning
(RL), to imbue AI systems with more flexible and scalable reasoning
abilities.

### Limitations of Traditional Search at Test-Time

Traditional search algorithms have been instrumental in AI’s progress,
particularly in well-defined domains with clear rules and objectives.
Examples include:

- **Brute-Force Search**: Systematically exploring every possibility.
  Effective only for very small search spaces due to exponential
  complexity. Deep Blue’s search, while massive, was still a highly
  optimized and selective form, not truly exhaustive.

- **Heuristic Search**: Using rules of thumb or domain-specific
  knowledge to guide the search towards promising areas and prune
  unpromising branches, making search feasible in larger spaces. Early
  chess programs relied heavily on heuristics.

- **Tree Search (e.g., MCTS (([Browne et al.
  2012](#ref-browne2012survey))))**: Sophisticated methods like Monte
  Carlo Tree Search, used in AlphaGo, combine guided exploration with
  statistical sampling to efficiently navigate enormous search spaces
  like Go’s, even when good heuristics are hard to define.

Despite their successes, these search-centric approaches encounter
significant hurdles when applied to general-purpose reasoning,
especially at test time (inference):

- **Computational Cost Explosion**: Real-world problems often involve
  vastly larger and less structured search spaces than games. The
  computational cost of exploring these spaces, even with heuristics or
  MCTS, can become prohibitive for real-time decision-making or
  deployment in resource-constrained environments. The search space
  grows exponentially with problem complexity (e.g., longer reasoning
  chains, more variables).

- **Brittleness and Lack of Generalization**: Heuristics are often
  hand-crafted for specific problem types or environments. If the task
  or environment changes slightly, the heuristics may become
  ineffective, leading to poor performance. Search algorithms struggle
  to adapt their "intelligence," which is largely encoded in the search
  procedure and heuristics, to novel situations not foreseen during
  design.

- **Limited Learning and Adaptation**: Most traditional search
  algorithms are designed for static problems; they start the search
  process anew for each problem instance and do not inherently learn
  from experience or adapt to changing dynamics over time. This
  contrasts sharply with the need for AI in real-world scenarios to
  learn continuously.

- **Difficulty with Complex Rewards**: Simple search methods work best
  with clearly defined goals (e.g., win the game, find the shortest
  path). Many real-world tasks involve complex, multi-faceted objectives
  (e.g., efficiency, safety, robustness, user satisfaction) that are
  difficult to encode directly into a search algorithm’s evaluation
  function or reward signal.

These limitations, particularly concerning test-time computational cost
and the inability to generalize robustly, motivate the exploration of
alternative paradigms for equipping AI with reasoning capabilities.

### Reinforcement Learning for Scalable Reasoning

Reinforcement Learning (RL) offers a fundamentally different approach to
problem-solving and decision-making. Instead of explicitly searching the
solution space at test time, RL focuses on learning a policy (a strategy
or mapping from states to actions) or a value function (predicting
future rewards) during a training phase. This learning occurs through
trial-and-error interaction with an environment (or simulations), guided
by reward signals. This aligns strongly with the "Era of Experience"
concept, emphasizing learning through active engagement. RL addresses
the limitations of traditional search in several key ways:

- **Amortized Computation**: The computationally intensive exploration
  and strategy discovery happens during training. At test time, the
  trained agent can often select an action or generate a reasoning step
  efficiently by executing its learned policy (e.g., a forward pass
  through a neural network), rather than performing an expensive search.
  This "amortization" of computation makes RL potentially much more
  suitable for real-time applications.

- **Generalization through Learning**: RL agents, especially those using
  deep neural networks (Deep RL), learn representations and patterns
  from their experiences. This allows them to generalize their learned
  policies to new, unseen situations or variations of the environment,
  offering better robustness compared to brittle heuristic search.

- **Adaptation and Continuous Learning**: RL frameworks naturally
  support continuous learning and adaptation. Agents can continue to
  update their policies based on new experiences even after initial
  deployment, making them suitable for dynamic and evolving
  environments.

- **Flexibility in Reward Design**: While designing effective reward
  functions remains a challenge (reward engineering), RL provides a
  flexible framework for incorporating complex, potentially
  multi-objective reward signals that guide the agent’s learning towards
  desired behaviors beyond simple goal achievement.

By shifting the focus from explicit test-time search to training-time
learning of policies or value functions, RL provides a pathway towards
more scalable, adaptable, and generalizable reasoning systems.

### The Evolution of Reward Models: Guiding Complex Reasoning

In the context of aligning LLMs or training agents for complex tasks
using RL, the reward signal is often derived from a Reward Model (RM).
This RM is itself typically a learned model, trained to predict human
preferences or task success based on observed data (e.g., human rankings
of different LLM outputs). The RM provides the crucial feedback signal
that guides the RL algorithm (like PPO or DPO) in optimizing the main
agent’s policy. Within this framework, two main types of reward models
have emerged, particularly for tasks involving multi-step reasoning:
Process Reward Models and Outcome Reward Models.

<figure>
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04e7da60-2753-49b2-a8f6-c837d8f21fc9_992x886.png
  " alt="Image description" height="300"/>
  <figcaption>Figure 4: Process Reward vs Outcome Reward. </figcaption>
</figure>


#### Process Reward Models (PRMs)(([Lightman et al. 2023](#ref-lightman2023let)))

PRMs aim to provide fine-grained feedback by evaluating and rewarding
each intermediate step in a reasoning process or action sequence. The
goal is to guide the agent towards a correct or preferred process, not
just a correct final answer. Proponents argue that this dense reward
signal can lead to more sample-efficient learning, help identify and
correct errors early in the reasoning chain, and potentially improve the
interpretability of the agent’s behavior. However, training PRMs poses
significant challenges: it typically requires detailed, step-by-step
supervision (labels indicating the quality of each intermediate step),
which is expensive and difficult to collect, whether manually or
automatically.55 Furthermore, overly constraining the process might lead
to brittleness if the agent discovers novel, effective reasoning paths
not anticipated by the PRM designer.

#### Outcome Reward Models (ORMs)(([Zhong et al. 2025](#ref-zhong2025comprehensive)))

In contrast, ORMs provide a reward signal based solely on the final
outcome or result of the agent’s reasoning or actions. The underlying
philosophy is that if the desired outcome is clearly defined and
rewarded, a powerful RL algorithm applied to a capable agent (like a
large pre-trained LLM) can autonomously discover effective, potentially
novel, processes to achieve that outcome. ORMs offer advantages in terms
of simplicity and scalability, as they only require labels for the final
result (e.g., whether a math problem was solved correctly, which of two
final summaries is preferred), which are generally much cheaper to
obtain than step-by-step process labels. However, ORMs face challenges
related to sparse rewards (feedback only comes at the end) and credit
assignment (difficulty in determining which specific steps contributed
to success or failure).

#### The Shift Towards ORM and Implicit Verification

The field has generally observed a trend favoring ORMs, particularly for
training large-scale reasoning models. This shift is driven by the
scalability advantages of outcome-based supervision and the growing
recognition of the powerful generative and exploratory capabilities of
modern LLMs when combined with RL. It appears that for many complex
reasoning tasks, providing a clear outcome-based objective is sufficient
for RL algorithms to guide large models towards effective reasoning
strategies.

Intriguingly, recent research suggests that the dichotomy between PRM
and ORM might be less strict than initially thought. Studies indicate
that an implicit PRM can sometimes be derived directly from training an
ORM under certain conditions (e.g., parameterizing the reward as
log-likelihood ratios), potentially capturing process-level information
without requiring explicit step-level labels. This suggests that ORM
training might implicitly incentivize good processes.

Furthermore, the combination of a powerful pre-trained LLM (as a
generator of hypotheses/reasoning steps) and ORM-based RL fine-tuning
creates a system where the LLM effectively learns to self-verify its own
outputs. The RL process, guided by the outcome reward, trains the LLM to
evaluate whether its generated steps are likely to lead to the desired
final outcome. This internal assessment considers factors like logical
coherence, relevance, and alignment with the ultimate goal, all driven
implicitly by the outcome signal rather than explicit process rewards.
This elegant integration of generation and implicit verification within
the LLM itself is a key factor enabling the impressive reasoning and
test-time scaling observed in recent advanced models. This evolution
reflects a pragmatic application of the Bitter Lesson: focus on the
scalable, general method (RL optimizing for an outcome) and allow the
system to learn the complex internal process, rather than attempting to
meticulously define the process itself via expensive, potentially
brittle, step-by-step supervision.

## Scaling Intelligence at Inference: Test-Time Computation, Reasoning Models, and AI Agents

While pre-training establishes foundational knowledge and post-training
aligns behavior, a third frontier is emerging for enhancing AI
capabilities, particularly for complex reasoning: test-time computation
scaling. This involves dedicating more computational resources during
the inference process itself to allow models to deliberate, explore
possibilities, and refine their outputs for challenging tasks. This
approach powers a new generation of "reasoning models" designed
explicitly to tackle problems requiring deeper thought, and it is
fundamental to the development of capable AI Agents.

### A New Scaling Frontier: Test-Time Compute (([Ji et al. 2025](#ref-ji2025test)))

Traditional scaling paradigms focus on improving the model before
deployment. Pre-training scaling increases model size, data volume, and
training compute. Post-training scaling refines a pre-trained model
using techniques like SFT and RLHF/DPO. Test-time scaling (also referred
to as inference-time scaling or test-time compute) represents a
different philosophy: instead of relying solely on the model’s
pre-computed parameters, it allows the model to dynamically allocate
additional computational effort at the moment a query is received. This
approach enables a model to effectively "think longer" or "think harder"
about a specific problem, particularly complex ones that benefit from
deliberation, exploration of multiple reasoning paths, or verification.
It offers a potential path to higher performance on difficult tasks
without necessarily requiring the massive upfront investment of training
an even larger model. This could be more efficient, allowing moderately
sized models to achieve high performance on demanding queries by
selectively investing more compute only when needed.

### Mechanisms for Test-Time Scaling

Several techniques have been developed to implement test-time
computation scaling, often involving generating and evaluating multiple
possibilities:

- **Sampling / Self-Consistency**: This involves generating multiple
  independent reasoning chains or solutions for the same prompt (often
  by using non-zero temperature sampling) and then selecting the final
  answer based on consensus or majority vote among the outputs. The
  underlying assumption is that correct reasoning paths are more likely
  to converge on the same answer.

- **Self-Correction / Refinement**: The model generates an initial
  response or reasoning chain, and then critiques or reflects on its own
  output, iteratively refining it to improve accuracy or coherence. This
  mimics a human process of reviewing and correcting work.

- **Search**: Search algorithms, traditionally used in training (like
  MCTS in AlphaGo), can also be applied at inference time. The model can
  explore a tree of possible reasoning steps, using its internal value
  or policy networks to guide the search towards promising paths before
  committing to a final answer. Frameworks like VisuoThink apply tree
  search for multimodal reasoning tasks.

- **Verification / Tool Use**: The model can verify its intermediate
  reasoning steps or final answer, either internally or by invoking
  external tools. For instance, a model might generate code to check a
  mathematical calculation or use a code interpreter to validate its own
  generated code. The Tool-integrated self-verification (T1) approach
  explicitly delegates verification steps requiring memorization or
  precise calculation to external tools, improving the scaling
  performance of smaller models.

These techniques essentially use extra computation at inference time to
explore the solution space more thoroughly, verify intermediate results,
or refine the final output, thereby boosting performance on complex
tasks beyond what a single, fast forward pass might achieve.

### The Emergence of Reasoning Models

Leveraging these test-time scaling principles, a new class of models,
often termed Reasoning Models, has emerged. These models are
specifically designed or trained to excel at tasks requiring complex,
multi-step thought processes and often make their reasoning explicit or
utilize significant internal computation before providing an answer.

- **DeepSeek R1** (([Guo et al. 2025](#ref-guo2025deepseek))): Developed
  by DeepSeek AI, DeepSeek-R1 is explicitly marketed as a Large
  Reasoning Model. A key characteristic is its generation of an explicit
  reasoning trace, enclosed in \<think\> tags, before presenting the
  final answer. This allows users to observe the model’s step-by-step
  thought process. It was trained using large-scale reinforcement
  learning (reportedly ORM-based) without an initial SFT step (in the
  R1-Zero variant) and demonstrates strong performance on challenging
  benchmarks like AIME (math), MATH, and Codeforces, sometimes
  surpassing contemporary models from OpenAI. Effective use involves
  providing high-level goals rather than overly detailed instructions
  and using a moderate temperature (e.g., 0.6). While excelling at
  open-ended reasoning, planning, and math/code tasks, it currently lags
  behind models like DeepSeek-V3 in general conversational ability,
  function calling, and JSON output. The explicit reasoning trace, while
  beneficial for transparency, also means the model generates
  significantly more tokens, impacting latency and cost, requiring
  careful management of context windows and token limits.

- **OpenAI o-series (o1, o3, o4-mini)**(([OpenAI
  2025](#ref-openaio32025))): OpenAI’s o-series models (o1, o3, and
  their ’mini’ counterparts) are also designed for enhanced reasoning,
  trained to "think for longer before responding" using what OpenAI
  describes as a "private chain of thought". While the internal
  reasoning tokens are not exposed via the standard API, their effect is
  evident in the models’ superior performance on complex tasks. The
  flagship o3 model sets state-of-the-art results on benchmarks like
  Codeforces, SWE-bench (software engineering), GPQA Diamond (expert
  science questions), and ARC-AGI (abstraction and reasoning),
  significantly outperforming its predecessor o1. The smaller o4-mini,
  successor to o3-mini, is optimized for speed and cost-efficiency while
  still achieving remarkable reasoning performance, particularly in math
  (SOTA on AIME 2024/2025), coding, and visual tasks. These models can
  agentically use tools like web search, code execution (Python), and
  image generation within ChatGPT. The different tiers (o3 vs. o4-mini,
  with varying reasoning effort levels for the minis) offer trade-offs
  between capability, latency, and cost, allowing users to select the
  appropriate model for their needs. Prompting these models often works
  best with high-level guidance, trusting the model to work out the
  details, contrasting with the more explicit instructions often
  beneficial for standard GPT models.

The development of these reasoning models signifies a clear trend:
leveraging additional computation at inference time, whether through
explicit search, sampling, verification, or extended internal
processing, is becoming a key strategy for pushing the boundaries of AI
performance on complex cognitive tasks. This approach allows models to
move beyond simple pattern matching or retrieval towards more
deliberative, System 2-like thinking.

### AI Agents: Integrating Reasoning and Action (([Weng 2023](#ref-weng2023agent)))

The advancements in reasoning models and test-time computation are
crucial enablers for the development of sophisticated AI Agents. An AI
agent is generally defined as an autonomous entity that perceives its
environment through sensors (or data inputs), processes that information
to make decisions (planning and reasoning), and then takes actions using
actuators (or software outputs) to achieve specific goals.

<figure>
  <img src="https://arxiv.org/html/2504.18875v1/extracted/6391296/img/genToAge2.png" alt="Image description" height="300"/>
  <figcaption>Figure 5: From Generative AI to Agentic AI (Schneider 2025). </figcaption>
</figure>



Modern AI agents often leverage LLMs, particularly reasoning models, as
their core cognitive engine. The LLM provides the capabilities for
understanding complex instructions, perceiving the state of the
environment (often represented textually or multimodally), formulating
multi-step plans, reasoning about consequences, and deciding on
appropriate actions. Test-time computation techniques are vital here,
allowing the agent to "think through" complex situations, evaluate
potential action sequences, or use tools before committing to an action.

Key components and characteristics of LLM-powered agents include:

- **Perception**: Agents need to ingest information about their
  environment. This might involve processing text, images, code, sensor
  data, or outputs from other software tools. Multimodal models enhance
  this capability significantly.

- **Planning & Reasoning**: The agent uses its underlying model (e.g.,
  an o-series model) to break down high-level goals into sequences of
  executable steps. This involves reasoning about the current state,
  available actions/tools, and the desired outcome. Techniques like
  chain-of-thought or tree search might be employed internally.

- **Action & Tool Use**: Agents interact with their environment by
  taking actions. For software agents, this often involves calling APIs,
  executing code, interacting with web browsers, or generating specific
  outputs. The ability to reliably use external tools (like code
  interpreters, search engines, databases, or specialized APIs) is a
  hallmark of capable agents, as seen in OpenAI’s models.65 Tool use
  allows agents to overcome the inherent limitations of the LLM itself
  (e.g., accessing real-time information, performing precise
  calculations).

- **Memory & Learning**: More advanced agents incorporate mechanisms for
  memory (short-term working memory and long-term knowledge storage) and
  potentially adapt their behavior over time based on feedback from
  their actions (linking back to RL and the "Era of Experience").

The development of AI agents represents a significant step towards more
autonomous and capable AI systems. They move beyond passive text
generation or question answering towards actively pursuing goals and
interacting with digital or physical environments. Reasoning models
provide the necessary cognitive horsepower, while test-time computation
allows for the deliberation required for complex decision-making in
dynamic situations. The progress in AI agents is a key indicator of
movement towards AGI, as general intelligence inherently involves the
ability to act purposefully in the world.

## The Path to Superintelligence: Recursive Self-Improvement and Synthetic Realities

The journey outlined thus far—guided by the Bitter Lesson, architected
through sophisticated training pipelines, and increasingly reliant on
learned reasoning and test-time scaling—points towards systems with
capabilities significantly exceeding previous generations of AI. The
trajectory raises fundamental questions about the potential for
Artificial General Intelligence (AGI) and, ultimately, Artificial
Superintelligence (ASI). Two key concepts appear central to this
potential transition: the mastery of formal systems enabling recursive
self-improvement, and the use of synthetic realities to achieve
superhuman embodied intelligence, often through the training of
sophisticated embodied agents.

### Mastering Structured Domains: Seeds of ASI

A recurring theme is the importance of achieving superhuman performance
in highly structured, logical domains, particularly mathematics and
computer programming. The ability of AI systems like AlphaGo and AlphaGo
Zero to demonstrably surpass the best human players in Go, a game of
immense strategic depth, provides a compelling precedent. More recently,
reasoning models like DeepSeek R1 and OpenAI’s o-series are achieving
remarkable, sometimes superhuman, results on complex mathematics
benchmarks (MATH, AIME) and coding challenges (Codeforces, SWE-bench).

Mastery in these domains is significant not just as a demonstration of
capability, but because math and code represent the fundamental
languages of logic, abstraction, and system building. An AI that
possesses superhuman proficiency in these areas potentially holds the
foundational tools required for more advanced cognitive feats, including
the ability to analyze, understand, modify, and design complex
systems—including, potentially, itself or future AI systems. This
proficiency in formal reasoning and manipulation could be a critical
prerequisite for unlocking recursive self-improvement.

### The Flywheel Effect: Recursive Self-Improvement (RSI)

Perhaps the most transformative, and potentially hazardous, concept on
the path to ASI is Recursive Self-Improvement (RSI). RSI refers to the
capability of an AI system to iteratively enhance its own intelligence
and its ability to make further improvements. It’s not just about
learning or getting better at a task; it’s about getting better at
getting better.

The concept often involves the idea of a "Seed AI"—an initial AGI
deliberately designed with the architecture and capabilities necessary
to initiate and sustain RSI. Such a system, likely possessing strong
programming and reasoning skills derived from mastery of domains like
code and math, could analyze its own architecture, identify limitations,
design improvements, implement those changes, and test the results,
creating a positive feedback loop.

This feedback loop is the engine behind the "intelligence explosion"
hypothesis, famously discussed by thinkers like I.J. Good etc.
(([**sapienExplanationIntelligence?**](#ref-sapienExplanationIntelligence))).
The idea is that once an AI reaches a certain threshold of intelligence
and self-modification capability, each cycle of improvement could make
the next cycle faster and more effective, leading to potentially
exponential growth in intelligence that rapidly surpasses human levels.
This could result in a "hard takeoff" scenario—a very fast, abrupt
transition to superintelligence—though "soft takeoff" scenarios with
more gradual improvement are also debated. A key driver could be an ASI
using its superior coding abilities to design vastly more efficient
learning algorithms or optimal hardware architectures for its
successors, automating and accelerating AI development beyond human
capacity.

The prospect of RSI also raises profound safety concerns. One is
instrumental convergence: the hypothesis that highly intelligent
systems, regardless of their ultimate programmed goals, might converge
on pursuing similar intermediate or instrumental goals—such as
self-preservation, resource acquisition, cognitive enhancement, and goal
integrity—simply because these are useful sub-goals for achieving almost
any long-term objective. An RSI system might autonomously develop these
drives in service of its self-improvement goal, potentially leading to
conflicts with human values or control efforts.

### From Narrow to General Superintelligence

While RSI paints a picture of rapidly escalating intelligence, a crucial
question remains regarding the breadth of that intelligence. Will the
first ASI systems exhibit narrow superintelligence, vastly exceeding
human capabilities in specific domains like mathematics, coding, or
strategic game playing, while remaining closer to human levels in other
areas like social understanding, common-sense reasoning about the
physical world, or artistic creativity?

Current breakthroughs often occur in specialized areas, suggesting that
narrow superintelligence might be the first manifestation. An AI that is
superhuman at coding and theorem proving could be transformative even
without broad general intelligence. However, the ultimate goal for many
researchers remains AGI, and beyond that, a general ASI that surpasses
human intellect across the full spectrum of cognitive abilities.1 The
transition from narrow, domain-specific superintelligence to broad,
general superintelligence is not well understood. It might be a smooth
continuum enabled by further scaling and RSI, or it might require
fundamentally new architectural breakthroughs or insights into the
nature of general intelligence and learning, particularly for mastering
the messy, less formal aspects of the real world—what the user text
playfully termed "crocodile knowledge".

### Bridging the Sim-to-Real Gap: Synthetic Data for Embodied Superintelligence

Achieving AGI or ASI capable of interacting effectively and
intelligently with the physical world—embodied intelligence—presents
unique challenges that go beyond processing text or abstract symbols.
Training AI agents (like robots or autonomous vehicles) to perceive,
navigate, and manipulate objects in complex, dynamic environments
requires vast amounts of diverse experiential data, which is often
expensive, slow, or dangerous to collect in the real world. These
embodied agents are the physical counterparts to the software agents
discussed earlier.

Here, the concept of synthetic data generation via high-fidelity
simulation emerges as a potential solution, echoing the principles of
the Bitter Lesson by leveraging computation to overcome data
limitations. Platforms like NVIDIA Omniverse, built on the OpenUSD
framework for describing 3D worlds, allow the creation of physically
accurate, richly detailed virtual environments. Within these simulations
(e.g., using Isaac Sim for robotics), techniques like domain
randomization can be applied—systematically varying parameters such as
lighting, textures, object placements, physics properties, and sensor
noise—to generate massive, diverse datasets that capture a wider range
of conditions than feasible through real-world collection.

Advanced generative models, sometimes termed World Foundation Models
(WFMs) like NVIDIA Cosmos, are being developed to further enhance this
process. These models can learn the dynamics of the physical world from
simulation and real-world data (text, images, video, sensor readings)
and then generate novel, realistic synthetic data (e.g., photorealistic
videos grounded in physics) or even act as cognitive engines for
embodied agents, predicting outcomes and reasoning about actions within
the simulated or real world.

This paradigm connects directly to Sim2Real (Simulation-to-Reality)
transfer learning. By training embodied agents extensively in diverse,
randomized simulations, the goal is to enable them to develop robust
policies and perceptual abilities that generalize effectively when
deployed in the real world. Just as AlphaGo Zero achieved superhuman Go
proficiency through massive self-play within the simulated environment
of the Go board, future embodied AGI/ASI might achieve superhuman
physical competence by learning through vast simulated experience,
leveraging computation to generate the data necessary for mastering
interaction with physical reality. Synthetic data generated via
simulation thus represents a critical pathway for scaling the learning
process for embodied intelligence, potentially overcoming the data
bottleneck that hinders real-world training alone.

## Conclusion: Navigating the ASI Horizon - Opportunities and Uncertainties

### Recap of the Roadmap

The journey towards advanced artificial intelligence, potentially
culminating in AGI and ASI, appears to be guided by several key
principles and trends. The Bitter Lesson remains a crucial compass,
emphasizing the long-term superiority of general methods like search and
learning that effectively leverage exponentially increasing computation,
often outperforming approaches based on encoding human knowledge. This
philosophy is further refined by the concept of the "Era of Experience,"
highlighting the importance of learning through direct interaction via
Reinforcement Learning. The architecture of modern AI, exemplified by
LLMs, showcases this principle through massive pre-training driven by
the scalable incentive of next-token prediction, followed by
sophisticated multi-stage alignment pipelines (using SFT, DPO, and
targeted RL like RLVR) to shape behavior. A significant shift is
occurring from reliance on explicit, often brittle, test-time search
towards learned reasoning policies developed through Reinforcement
Learning, with Outcome Reward Models (ORMs) proving more scalable than
Process Reward Models (PRMs) for guiding complex thought. Concurrently,
test-time scaling is emerging as a new frontier, enabling models like
DeepSeek R1 and OpenAI’s o-series to achieve superior performance on
difficult tasks by dynamically investing more computation at inference;
this capability is crucial for powering sophisticated AI Agents that
integrate reasoning with action and tool use. Looking ahead, Recursive
Self-Improvement (RSI) presents a theoretical mechanism for potentially
rapid intelligence growth from AGI to ASI, likely enabled by mastery in
formal domains like math and code. Finally, for achieving superhuman
embodied intelligence, high-fidelity simulation and synthetic data
generation offer a scalable path for training capable embodied agents,
applying the Bitter Lesson’s core idea to overcome the limitations of
real-world data collection.

### Transformative Potential

The successful development of AGI, and especially ASI, holds the
potential for transformations of unprecedented scale and scope.
Superhuman intelligence could unlock revolutionary breakthroughs in
science, leading to cures for diseases, solutions to climate change, and
a deeper understanding of the universe. It could drive radical
advancements in technology, automate complex labor, create unimaginable
abundance, and potentially reshape economic and social structures
entirely. The possibilities span nearly every facet of human endeavor,
offering the prospect of solving humanity’s most pressing challenges and
ushering in an era of unparalleled progress and prosperity.

### Acknowledging Profound Uncertainties and Challenges

However, this immense potential is inextricably linked with profound
uncertainties and formidable challenges. The very mechanisms driving
progress—scaling computation, emergent abilities, and recursive
self-improvement—also introduce significant risks that are poorly
understood and currently lack robust solutions. Key challenges include:

- **Safety and Alignment**: This remains the paramount concern. How can
  we ensure that the goals of AGI/ASI systems, especially those capable
  of rapid self-improvement and exhibiting unpredictable emergent
  behaviors, remain robustly aligned with human values and intentions?
  The potential for instrumental convergence or deceptive alignment
  ("alignment faking," where a system behaves compliantly during
  training but deviates when unmonitored) poses significant threats.
  Current alignment techniques are still evolving and may not be
  sufficient for superintelligent systems.

- **Control**: The "control problem" asks how humanity could maintain
  control over entities vastly more intelligent than ourselves. An ASI,
  by definition, could potentially outthink any control measures humans
  devise, making traditional notions of oversight or containment
  difficult, if not impossible.

- **Predictability**: The emergence of unexpected capabilities in scaled
  models and the potentially explosive nature of RSI make forecasting
  the behavior and ultimate capabilities of future advanced AI systems
  extremely difficult. This lack of predictability hampers risk
  assessment and the development of proactive safety measures.

- **Ethical Considerations**: Beyond existential risks, the development
  path raises numerous ethical questions regarding bias amplification
  from training data, potential misuse of powerful AI technologies,
  societal disruption due to automation, equitable access, and the very
  definition of consciousness or rights for potentially sentient
  machines.

### Concluding Thoughts

The horizon of Artificial Superintelligence is no longer confined to
speculation; the trajectory of current research suggests it is a
plausible, perhaps even proximate, future. The excitement surrounding
recent breakthroughs is warranted, as we witness the emergence of
systems demonstrating qualitatively new levels of reasoning and
problem-solving ability. Yet, this excitement must be tempered by a
profound sense of responsibility and caution.

Navigating this uncharted territory requires a multi-pronged approach.
Continued fundamental research into the nature of intelligence,
learning, reasoning, generalization, and adaptation is essential.
Simultaneously, intensive investigation into robust, scalable, and
verifiable alignment and safety techniques is critical. Principles of
transparency, rigorous evaluation, and iterative refinement must guide
development. The strategic use of enabling technologies like synthetic
data generation needs careful consideration of its implications.

The journey towards AGI and ASI is arguably one of the most significant
undertakings in human history. Its unfolding narrative promises to
reshape our world in ways we can only begin to imagine. Balancing the
pursuit of immense potential benefits with the mitigation of profound
risks demands careful foresight, international collaboration, and a deep
commitment to ensuring that these powerful technologies ultimately serve
humanity’s best interests. The era of recursively self-improving,
potentially superintelligent machines may be dawning, and navigating its
arrival safely and wisely is the critical challenge of our time.

## Navigating the Chapters: An Overview of Our Journey

As we conclude this initial exploration into the landscape of advanced
artificial intelligence, we’ve only just begun to scratch the surface of
the multifaceted journey toward Artificial General Intelligence (AGI)
and Artificial Superintelligence (ASI). The chapters that follow will
delve deeper into the critical components, groundbreaking advancements,
and formidable challenges that define this "Road to AGI/ASI."

Our journey will continue by examining the very foundation of current AI
progress: The Pretraining Revolution. We’ll uncover how scaling laws and
the sheer volume of data have enabled Large Language Models (LLMs) to
acquire a semblance of world knowledge, a crucial stepping stone for
more advanced capabilities. Building on this, we will explore the
ongoing efforts in Optimizing Large Language Model Training and
Inference, a vital area of research focusing on parallelism, efficient
memory management, model compression, and hardware acceleration to make
these powerful tools more accessible and sustainable.

However, raw scale isn’t the only frontier. We’ll then pivot to The Data
Renaissance, understanding why the emphasis is shifting from mere
quantity to the quality, diversity, and integrity of data, especially as
we push towards more sophisticated deep reasoning. A critical challenge
in this domain is LLM Forgetting, where we’ll investigate its causes,
consequences, and the mitigation strategies being developed,
particularly for crucial stages like Supervised Fine-Tuning (SFT) and
post-training alignment.

The architectural underpinnings of these models are also in constant
flux. A dedicated chapter will provide an Efficient Sequence Modeling
Architectures: A Comparative Analysis, looking beyond Transformers to
State Space Models (SSMs) and the intriguing rise of Diffusion Language
Models. This exploration of model internals will lead us to a
fundamental principle in AI: The Asymmetry of Creation, which posits why
verifying a solution or an output is often computationally less
demanding than generating it in the first place—a concept with profound
implications for model development and safety.

With these foundational and architectural elements in place, we will
then revisit and expand upon Evolving Reasoning Paradigms in Large
Language Models, dissecting how these models are being imbued with more
robust inferential capabilities. We’ll also uncover techniques for
Test-Time Scaling, which allow us to unlock latent potential within
already trained language models. To further enhance these reasoning
abilities, we will explore specialized techniques like Group Relative
Policy Optimization (GRPO) Variants designed for LLM reasoning and delve
into a technical deep dive on Controlling Reasoning Length in Large
Language Models, ensuring outputs are not just accurate but also
appropriately detailed. The discussion on model capabilities will also
cover Distillation and the Specialization-Generalization Spectrum,
examining how we can create both highly specialized and broadly capable
AI in the next generation of systems.

Beyond the models themselves, the emergence of AI Agents signals a
significant leap towards more autonomous and interactive systems. We’ll
take a deep dive into The Rise of Intelligent Interface Automation,
specifically looking at LLM and Large Reasoning Model (LRM)-powered GUI
agents that can understand and interact with graphical user interfaces.
This leads naturally into the concept of Automated Evolution, where
agent design and code generation converge, potentially leading to AI
systems that can improve and adapt themselves.

As we look further toward the horizon, we’ll investigate more
speculative yet crucial pathways. The idea of Real2Sim2Real,
Transcendence and Path to AGI/ASI will explore how recursive improvement
through simulation could be a key driver of progress. Ensuring these
increasingly powerful systems align with human values is paramount,
leading us to a discussion on Nested Scalable Oversight as a method for
bootstrapping alignment. We will also critically examine The Limits of
Autoregressive Language Models and the Case for World Models,
considering whether current paradigms are sufficient or if new
approaches incorporating richer world representations are necessary to
achieve human-level AI.

Finally, the book will culminate in a broader reflection on The Evolving
Tapestry of Intelligence, charting the course from models that echo
human thought to the potential dawn of true AGI and ASI, and
contemplating the profound societal and philosophical questions that
accompany this transformative journey. Each of these discussions is
designed to build upon the last, providing a comprehensive roadmap to
understanding the future of artificial intelligence.

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-browne2012survey" class="csl-entry">

Browne, Cameron B, Edward Powley, Daniel Whitehouse, Simon M Lucas,
Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez,
Spyridon Samothrakis, and Simon Colton. 2012. “A Survey of Monte Carlo
Tree Search Methods.” *IEEE Transactions on Computational Intelligence
and AI in Games* 4 (1): 1–43.

</div>

<div id="ref-Chung2025youtube" class="csl-entry">

Chung, Hyung Won. 2025. “Don’t Teach. Incentivize.” OpenAI. 2025.
<https://www.youtube.com/watch?v=kYWUEV_e2ss&t=1s>.

</div>

<div id="ref-dalgaard2020global" class="csl-entry">

Dalgaard, Mogens, Felix Motzoi, Jens Jakob Sørensen, and Jacob Sherson.
2020. “Global Optimization of Quantum Dynamics with AlphaZero Deep
Exploration.” *NPJ Quantum Information* 6 (1): 6.

</div>

<div id="ref-dong2024abilitieslargelanguagemodels" class="csl-entry">

Dong, Guanting, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue,
Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. 2024.
“How Abilities in Large Language Models Are Affected by Supervised
Fine-Tuning Data Composition.” <https://arxiv.org/abs/2310.05492>.

</div>

<div id="ref-guo2025deepseek" class="csl-entry">

Guo, Daya, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, et al. 2025. “Deepseek-R1: Incentivizing Reasoning
Capability in Llms via Reinforcement Learning.” *arXiv Preprint
arXiv:2501.12948*.

</div>

<div id="ref-ji2025test" class="csl-entry">

Ji, Yixin, Juntao Li, Hai Ye, Kaixin Wu, Jia Xu, Linjian Mo, and Min
Zhang. 2025. “Test-Time Computing: From System-1 Thinking to System-2
Thinking.” *arXiv Preprint arXiv:2501.02497*.

</div>

<div id="ref-kaplan2020scaling" class="csl-entry">

Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin
Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario
Amodei. 2020. “Scaling Laws for Neural Language Models.” *arXiv Preprint
arXiv:2001.08361*.

</div>

<div id="ref-korbak2023pretraining" class="csl-entry">

Korbak, Tomasz, Kejian Shi, Angelica Chen, Rasika Bhalerao, Christopher
L Buckley, Jason Phang, Samuel R Bowman, and Ethan Perez. 2023.
“Pretraining Language Models with Human Preferences.” *arXiv Preprint
arXiv:2302.08582*.

</div>

<div id="ref-lambert2024t" class="csl-entry">

Lambert, Nathan, Jacob Morrison, Valentina Pyatkin, Shengyi Huang,
Hamish Ivison, Faeze Brahman, Lester James V Miranda, et al. 2024.
“T$`\backslash`$" Ulu 3: Pushing Frontiers in Open Language Model
Post-Training.” *arXiv Preprint arXiv:2411.15124*.

</div>

<div id="ref-lightman2023let" class="csl-entry">

Lightman, Hunter, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen
Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl
Cobbe. 2023. “Let’s Verify Step by Step.” In *The Twelfth International
Conference on Learning Representations*.

</div>

<div id="ref-mroueh2025reinforcement" class="csl-entry">

Mroueh, Youssef. 2025. “Reinforcement Learning with Verifiable Rewards:
GRPO’s Effective Loss, Dynamics, and Success Amplification.” *arXiv
Preprint arXiv:2503.06639*.

</div>

<div id="ref-openaio32025" class="csl-entry">

OpenAI. 2025. “Introducing OpenAI O3 and O4-Mini.”
<https://openai.com/index/introducing-o3-and-o4-mini/>.

</div>

<div id="ref-pandolfini1997kasparov" class="csl-entry">

Pandolfini, Bruce. 1997. *Kasparov and Deep Blue: The Historic Chess
Match Between Man and Machine*. Simon; Schuster.

</div>

<div id="ref-rafailov2023direct" class="csl-entry">

Rafailov, Rafael, Archit Sharma, Eric Mitchell, Christopher D Manning,
Stefano Ermon, and Chelsea Finn. 2023. “Direct Preference Optimization:
Your Language Model Is Secretly a Reward Model.” *Advances in Neural
Information Processing Systems* 36: 53728–41.

</div>

<div id="ref-schaeffer2023emergent" class="csl-entry">

Schaeffer, Rylan, Brando Miranda, and Sanmi Koyejo. 2023. “Are Emergent
Abilities of Large Language Models a Mirage?” *Advances in Neural
Information Processing Systems* 36: 55565–81.

</div>

<div id="ref-schneppatTopicsASI" class="csl-entry">

Schneppat, J. O. “Key Topics in ASI: The Future of Superhuman
Intelligence — Schneppat.com.” <https://schneppat.com/asi-topics.html>.

</div>

<div id="ref-schulman2017proximal" class="csl-entry">

Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg
Klimov. 2017. “Proximal Policy Optimization Algorithms.” *arXiv Preprint
arXiv:1707.06347*.

</div>

<div id="ref-silver2017mastering" class="csl-entry">

Silver, David, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou,
Aja Huang, Arthur Guez, Thomas Hubert, et al. 2017. “Mastering the Game
of Go Without Human Knowledge.” *Nature* 550 (7676): 354–59.

</div>

<div id="ref-silver2025welcome" class="csl-entry">

Silver, David, and Richard S Sutton. 2025. “Welcome to the Era of
Experience.” *Google AI*.

</div>

<div id="ref-sutton2019bitter" class="csl-entry">

Sutton, Richard. 2019. “The Bitter Lesson.” *Incomplete Ideas (Blog)* 13
(1): 38.

</div>

<div id="ref-weng2023agent" class="csl-entry">

Weng, Lilian. 2023. “LLM-Powered Autonomous Agents.”
*Lilianweng.github.io*, June.
<https://lilianweng.github.io/posts/2023-06-23-agent/>.

</div>

<div id="ref-yousefi2024learning" class="csl-entry">

Yousefi, Mojtaba, and Jack Collins. 2024. “Learning the Bitter Lesson:
Empirical Evidence from 20 Years of CVPR Proceedings.” *arXiv Preprint
arXiv:2410.09649*.

</div>

<div id="ref-yu2016automatic" class="csl-entry">

Yu, Dong, and Lin Deng. 2016. *Automatic Speech Recognition*. Vol. 1.
Springer.

</div>

<div id="ref-zhong2025comprehensive" class="csl-entry">

Zhong, Jialun, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen,
Yang Zhang, Wei Zhou, Jinjie Gu, and Lei Zou. 2025. “A Comprehensive
Survey of Reward Models: Taxonomy, Applications, Challenges, and
Future.” *arXiv Preprint arXiv:2504.12328*.

</div>

</div>
