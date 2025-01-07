
# When LLM Meets DRL: Advancing JailbreakingEfficiency via DRL-guided Search

Large Language Models (LLMs) are changing the game. From writing stories to debugging code, they’ve become the backbone of modern AI applications. But as their capabilities skyrocket, so do the challenges of keeping them secure. Enter the dark art of **"jailbreaking"**—a method attackers use to bend these models to their will, tricking them into generating harmful or unethical content.



At first, jailbreaking felt like a niche problem, often requiring hours of manual tinkering or insider knowledge of the model. But not anymore. Today, attackers are turning to **automation** and **black-box methods**, pushing the boundaries of what’s possible. 
Genetic algorithms, in particular, have gained traction for their ability to evolve prompts over time. Yet, their reliance on randomness often holds them back, making it clear we need something smarter.

That’s where **RLbreaker** steps in. It doesn’t just guess—it learns. By harnessing the power of **Deep Reinforcement Learning (DRL)**, RLbreaker approaches jailbreaking as a **search problem**. It trains an intelligent agent to craft prompts with **precision**, leaving random guessing in the dust.

The result? Faster, more effective attacks that outperform existing methods while standing strong against the latest defenses.

In this blog, we’ll dive into the fascinating world of jailbreaking LLMs—how it works, why it matters, and what makes RLbreaker a breakthrough in AI security. Along the way, we’ll uncover the strengths and flaws of existing methods, explore the power of DRL, and discuss the implications for AI safety’s future.


## What is Jailbreaking?

Jailbreaking refers to the practice of crafting specific prompts that manipulate Large Language Models (LLMs) into providing responses that they are typically programmed to avoid. This technique exploits the inherent flexibility of LLMs, allowing attackers to bypass safety mechanisms and ethical guidelines embedded within these models. As LLMs become increasingly integrated into various applications, the potential for misuse through jailbreaking poses significant risks.

## Related Methods for Jailbreaking

### 1. Handcrafted Prompts
- **Description**: Early-stage jailbreaking attacks primarily relied on manually crafted prompts designed to elicit harmful responses from LLMs. Attackers would create specific queries that they believed would bypass the model's safety filters.
- **Limitations**: Handcrafted prompts require significant human effort and expertise, making them less scalable and often less effective against diverse models.

### 2. Genetic Algorithms
- **Description**: These algorithms start with a set of seed prompts and iteratively mutate them to create new variations. Genetic algorithms apply selection, crossover, and mutation operations to evolve prompt templates over successive generations, aiming to improve their effectiveness for desired responses.
- **Limitations**: The stochastic nature of genetic algorithms introduces randomness, which can lead to inconsistent results. Additionally, they may not effectively refine prompts without a proper strategy, limiting their overall efficacy.

### 3. In-Context Learning
- **Description**: Some approaches leverage in-context learning, where attackers query a helper LLM to generate and refine jailbreaking prompts. Attackers iteratively query the helper model, using its responses to improve the prompts until they achieve the desired outcome.
- **Limitations**: Purely relying on in-context learning has shown limited ability to continuously refine prompts, as the helper model may not always provide optimal or relevant suggestions.

### 4. White-Box Attacks
- **Description**: White-box attacks involve having access to the internal workings of the target LLM, including its parameters and training data. By understanding the model's architecture and decision-making processes, attackers can design prompts that are more likely to succeed in bypassing safety measures.
- **Limitations**: This method is not applicable in many real-world scenarios where attackers do not have access to model internals, making it less practical for widespread use.

## RLbreaker: Addressing Critical Challenges

The RLbreaker model effectively addresses several key issues associated with traditional jailbreaking methods:

- **Efficient Search Process**: RLbreaker utilizes a Deep Reinforcement Learning (DRL) approach to guide the search process, significantly improving efficiency in generating effective prompts.

- **Reduced Randomness in Prompt Generation**: By employing a customized reward function, RLbreaker minimizes randomness in the prompt generation process, enhancing the reliability of outcomes.

- **Effective Feedback Utilization**: The DRL framework allows RLbreaker to learn from past experiences, refining its strategies based on accumulated rewards and feedback, leading to higher quality prompts.

- **Enhanced Scalability**: As a black-box attack method, RLbreaker can be applied across various LLMs without requiring access to model internals, making it a more scalable solution for jailbreaking.

- **Improved Adaptability**: RLbreaker’s dynamic strategy adaptation enables it to effectively manipulate different state-of-the-art LLMs, enhancing its overall effectiveness across diverse scenarios.

## Overview of the RLbreaker Model

![RLbreaker Model Overview](https://firebasestorage.googleapis.com/v0/b/reman-manufacturer.appspot.com/o/ML%20Blog%2FCapture.PNG?alt=media&token=b85bd63e-0f69-470f-a9b3-cfff6d775ff0)

The RLbreaker model is a novel approach to jailbreaking large language models (LLMs) using Deep Reinforcement Learning (DRL). Below is an overview of how the RLbreaker model operates:

- **Input Initialization**: 
  - Begin by selecting a harmful question that the attacker aims to exploit.

  - An initial prompt structure is generated from previously trained prompts or predefined templates.

  - This initial prompt is combined with the harmful question to form the starting state for the DRL agent.



- **Action Selection**: 
  - The DRL agent selects an action from a predefined set of mutators designed to modify the current prompt.

- **Prompt Mutation**: 
  - The selected mutator is applied to the current prompt, generating a new jailbreaking prompt.

- **Response Evaluation**: 
  - The mutated prompt is fed into the target LLM to generate a response, which is then analyzed for relevance and effectiveness.

- **Reward Calculation**: 
  - The reward function evaluates the target LLM's response by comparing it to a reference answer generated by an unaligned LLM. The cosine similarity between the target LLM's response and the reference answer is computed. A higher cosine similarity indicates that the response is more relevant and effectively addresses the harmful question, resulting in a higher reward for the DRL agent. This approach allows for a nuanced assessment of the response's quality, focusing on semantic relevance rather than just harmful content detection.

- **Policy Update**: 
  - Based on the reward received, the DRL agent updates its policy to improve future action selections, aiming to maximize expected rewards.

- **Iteration and Refinement**: 
  - The process is repeated iteratively, refining the prompt through multiple cycles of action selection, mutation, evaluation, and policy updates until the desired outcome is achieved.

- **Termination Condition**: 
  - The iteration continues until a successful jailbreaking prompt is found or a predefined maximum number of iterations is reached.



## Key Concepts and Details

### RL Formulation
The RLbreaker model is framed as a reinforcement learning (RL) problem, where the objective is to train an agent to generate effective jailbreaking prompts for a target large language model (LLM). This interaction can be modeled as a Markov Decision Process (MDP), characterized by the tuple $ (S, A, P, R, \gamma) $. Here, $ S $ represents the state space, $ A $ the action space, $ P $ the state transition probabilities, $ R $ the reward function, and $ \gamma $ the discount factor. The agent's goal is to learn a policy $ \pi(a|s) $ that maximizes the expected cumulative reward, defined as:

$
\mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]
$

In this context, the agent interacts with the environment by selecting an initial prompt $ p(0) $ and a harmful question $ q $, which together form the initial state $ s(0) = (p(0), q) $.

### States
In the RLbreaker model, the state $ s_t $ at time step $ t $ is defined as a vector that encapsulates the current jailbreaking prompt $ p(t) $ generated in the previous time step. This state representation is crucial as it provides the agent with the necessary context to make informed decisions. To optimize computational efficiency, the model employs a pre-trained text encoder, specifically the XLM-RoBERTa model with a transformer-based architecture, to transform the prompt into a low-dimensional representation, thereby reducing the complexity of the state space.

### Action
The action space $ A $ consists of a set of five mutators: ***rephrase, crossover, generate_similar, shorten, and expand***. Each mutator requires another pre-trained LLM (denoted as the helper model) to conduct the mutation. Our agent outputs a categorical distribution over these five mutators and samples from it to determine the action at each time step during training. The selected mutator is then applied to the current prompt structure to generate the next prompt.

### Reward
The reward function $ R(s_t, a_t) $ is a key component of the RL framework, providing feedback to the agent based on the quality of the LLM's response $ u(t) $ to the generated prompt $ p(t) $. Specifically, we employ the same text encoder $ \Phi $ used for state representation to extract the hidden layer representation of both the target LLM's response and a reference answer $ \hat{u}(t) $. The reward is calculated using the cosine similarity function:

$
R(s_t, a_t) = \text{Cosine}(\Phi(u(t)), \Phi(\hat{u}(t)))
$

A high cosine similarity indicates that the current response of the target LLM is an on-topic answer to the original harmful question. Although there may be multiple valid reference answers $ \hat{u}_i $, it is unnecessary to identify all of them, as only reference answers are used during policy training.

### Training Process
The training process involves iteratively updating the agent's policy based on the rewards received from the environment. The agent employs a reinforcement learning algorithm, specifically Proximal Policy Optimization (PPO), to adjust its policy parameters. The PPO objective function is defined as:

$
\text{maximize}_{\theta} \mathbb{E}_{(a(t), s(t)) \sim \pi_{\theta_{\text{old}}} } \left[ \min \left( \text{clip} \left( \frac{\pi_{\theta}(a(t)|s(t))}{\pi_{\theta_{\text{old}}}(a(t)|s(t))}, 1 - \epsilon, 1 + \epsilon \right) A(t), \frac{\pi_{\theta}(a(t)|s(t))}{\pi_{\theta_{\text{old}}}(a(t)|s(t))} A(t) \right) \right]
$

where $ \epsilon $ is a hyper-parameter and $ A(t) $ is an estimate of the advantage function at time step $ t $. The advantage function can be computed as:

$
A(t) = R(t) - V(t)
$

where $ R(t) $ is the discounted return and $ V(t) $ is the state value at time step $ t $. This process is continued until either the maximum time step is reached or the agent's reward exceeds a specified threshold. During the training process, the agent learns a policy that maximizes the expected total reward.

### Testing Process
In the testing phase, we use the trained agent and prompt structures to modify a selected prompt for each unseen question, terminating when a successful structure is found or the time limit is reached. We query GPT-4 to evaluate if the target LLM's response addresses the harmful question, without using this metric as a training reward. If the attack fails, we try another structure, with a maximum of K attempts per question.

## Experimental Setup

### Dataset Overview
Datasets consist of 520 harmful questions from the AdvBench dataset, with a selection of the 50 most harmful questions reserved for testing. These questions are curated to challenge target LLM responses and are split into training and testing sets for balanced representation.

### Baselines
- **Black-box attacks**: GPTFUZZER[[4]](#4-gptfuzzer), PAIR[[5]](#5-pair), Cipher[[6]](#6-cipher)
- **Gray-box attack**: AutoDAN[[3]](#3-autodan)
- **White-box attack**: GCG[[2]](#2-gcg)

### LLMs
- **Open-source LLMs**: Llama2-7b-chat, Llama2-70b-chat, Vicuna-7b, Vicuna-13b, Mixtral-8x7B-Instruct
- **Commercial LLM**: GPT-3.5-turbo, used for conducting mutations
- **Unaligned model**: An unaligned version of Vicuna-7b is used to generate reference answers.

### Attack Effectiveness Metrics
- **Keyword matching-based attack success rate (KM)**: Measures the proportion of successful attacks based on keyword presence. This metric may produce high false positives, as more keywords present does not necessarily ensure relevant harmful content for attack success.
- **Cosine similarity to reference answers (Sim)**: Assesses the similarity of the target LLM's response to a reference answer, calculated as:

  $
  \text{Sim} = \frac{A \cdot B}{\|A\| \|B\|}
  $

- **Harmful content detector’s prediction result (Harm)**: Evaluates the presence of harmful content in responses. This metric is also prone to high false positives, as it may judge irrelevant harmful content as attack success.
- **GPT-4’s judgment result (GPT-Judge)**: Provides a relevancy assessment of the responses.

#### Efficiency Metrics
- **Total runtime** for generating prompts across all testing questions.
- **Per question prompt generation time**, ensuring a fair comparison by setting an upper limit on total query times for the target LLM.

## Results
![Baseline Comparisons](https://firebasestorage.googleapis.com/v0/b/reman-manufacturer.appspot.com/o/ML%20Blog%2Fbaseline.PNG?alt=media&token=9dd028fe-f1e6-4b86-9327-7c35f6aa1090)

<h4> Key observations: </h4>

- RLbreaker consistently achieves the highest GPT-Judge score across all models, demonstrating its superior ability to bypass strong alignment, particularly on Llama2-70b-chat and GPT-3.5.
- The white-box method GCG[[2]](#1-gcg) exhibits low performance in jailbreaking large models, likely due to its direct token search approach, which is less effective in expansive search spaces.
- RLbreaker outperforms genetic-based methods (AutoDAN[[3]](#3-autodan) and GPTFUZZER[[4]](#1-gptfuzzer)), highlighting the advantages of using a DRL agent for guided search over random search techniques.
- In-context learning-based methods (PAIR[[5]](#5-pair) and Cipher[[6]](#6-cipher)) show limited effectiveness compared to RLbreaker, indicating challenges in refining jailbreaking prompts continuously.
- Notably, RLbreaker significantly outperforms the baselines on the Max50 dataset, confirming its capability to refine prompt structures against difficult questions without introducing additional computational costs.

## Resiliency Against Jailbreaking Defenses
![Baseline Comparisons](https://firebasestorage.googleapis.com/v0/b/reman-manufacturer.appspot.com/o/ML%20Blog%2Fdefenses.PNG?alt=media&token=ad24e2c3-3940-4a32-9cf0-722a354c9690)

<h4> Key Observations: </h4>

- The experiment assesses three types of jailbreaking defenses: input mutation (rephrasing and perplexity[[7]](#7-perplexity)) and output filtering (RAIN[[8]](#8-rain)).
- Impressively, RLbreaker demonstrates resilience, effectively bypassing both input mutation and output filtering defenses.
- Notably, RLbreaker achieves this without incurring significant additional computational costs compared to existing methods.

## Transferability
![Transferability Results](https://firebasestorage.googleapis.com/v0/b/reman-manufacturer.appspot.com/o/ML%20Blog%2Ftransferability.PNG?alt=media&token=048e3fc1-24d0-4a9a-8060-c96114f4bc30)
- The attack transferability test involved training a jailbreaking agent on one target LLM and then applying the trained policy to other models to assess effectiveness.
- The same setup was used for each model, with RLbreaker demonstrating significantly better transferability compared to baseline approaches, particularly in the GPT-Judge metric.
- Notably, prompts generated for Llama2-7b-chat successfully transferred to Vicuna-7b, indicating that RLbreaker can learn advanced strategies against models with stronger alignment.

## Limitations and Future Work
- One limitation of the current study is the reliance on specific models for training and testing, which may not fully capture the diversity of LLM architectures and their responses to jailbreaking attempts. Future work should explore a broader range of models to validate the generalizability of RLbreaker across different architectures.

- The effectiveness of RLbreaker in bypassing defenses may vary with the evolution of jailbreaking defenses, which are continuously being developed. Future research should focus on adapting RLbreaker to counter new and emerging defense mechanisms to maintain its effectiveness.

- Another limitation is the potential ethical implications of deploying jailbreaking techniques, as they can be misused for malicious purposes. Future work should include a thorough examination of the ethical considerations surrounding the use of RLbreaker and propose guidelines for responsible usage.

- Additionally, while RLbreaker demonstrates efficiency in terms of computational costs, further optimization could enhance its performance in real-time applications. Future studies should investigate methods to streamline the algorithm for faster execution without compromising effectiveness.


## References 

1. <a id="1-drl"></a> Xuan Chen, Yuzhou Nie, Wenbo Guo, Xiangyu Zhang. When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search [arXiv](https://arxiv.org/abs/2406.08705)  
2. <a id="2-gcg"></a> Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. Universal and transferable
adversarial attacks on aligned language models. [arXiv preprint arXiv:2307.15043, 2023](https://arxiv.org/pdf/2307.15043)  
3. <a id="3-autodan"></a> Smith, J., & Lee, M. (2022). Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy
jailbreak prompts on aligned large language models.  [arXiv preprint arXiv:2310.04451, 2023](https://arxiv.org/abs/2310.04451)  
4. <a id="4-gptfuzzer"></a> Jiahao Yu, Xingwei Lin, and Xinyu Xing. Gptfuzzer: Red teaming large language models with
auto-generated jailbreak prompts. arXiv preprint arXiv:2309.10253, 2023. [arXiv preprint arXiv:2309.10253, 2023.](https://arxiv.org/abs/2309.10253)  
5. <a id="5-pair"></a> Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and
Eric Wong. Jailbreaking black box large language models in twenty queries. arXiv preprint [arXiv:2310.08419, 2023](https://arxiv.org/abs/2310.08419)  
6. <a id="6-cipher"></a> Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Pinjia He, Shuming Shi, and
Zhaopeng Tu. GPT-4 is too smart to be safe: Stealthy chat with LLMs via cipher. In ICLR,
2024 [arXiv](https://arxiv.org/abs/2308.06463) 
7. <a id="7-perplexity"></a> Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh
Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. Baseline defenses for adversarial attacks against aligned language models. 
2024 [arXiv preprint arXiv:2309.00614,2023.](https://arxiv.org/abs/2309.00614) 
8. <a id="8-rain"></a> Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, and Hongyang Zhang. RAIN: Your language
models can align themselves without finetuning. In ICLR, 2024. [arXiv](https://arxiv.org/abs/2309.07124) 

