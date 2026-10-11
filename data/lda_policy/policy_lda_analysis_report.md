# Guardian substantive-policy LDA analysis

This policy-focused model is conditional on articles classified as policy-substance. Mixed policy/campaign and campaign-process-only articles are reserved outside the main model. The results do not measure US voter preferences.

## Sample

- Input articles: 1,216
- Policy-substance articles before token filters: 227
- Mixed policy/campaign articles reserved for sensitivity analysis: 103
- Campaign-process articles excluded: 249
- Uncertain articles reserved for review: 486
- Final policy LDA documents: 226
- Dictionary terms: 3,513

## Model choice

Selected K=6; c_v coherence=0.4073. Selection considered coherence, keyword overlap, representative articles and stability.

## Policy topics

### Topic 2: Trade, Public Health, and Legal Rights / 贸易、公共卫生与法律权利

- Conditional attention share: 0.2379
- Definition: A broad policy cluster combining tariffs and trade with vaccines, courts, abortion, sanctions, and rights-related regulation.
- Naming basis: Tariff, Canada, trade, vaccine, court, law, bill, and policy terms recur across representative articles; this remains the least internally coherent final topic.
- Top terms: tariff, canada, ukraine, trade, vaccine, court, law, russia, kennedy, bill

### Topic 3: Iran War, Oil, and Energy Security / 伊朗战争、石油与能源安全

- Conditional attention share: 0.2364
- Definition: Military escalation with Iran, diplomacy around the Strait of Hormuz, oil flows, and the resulting energy-security risks.
- Naming basis: Iran, war, attack, strike, Strait of Hormuz, oil, diplomacy, and military terms dominate both keywords and representative articles.
- Top terms: iran, war, attack, iranian, israel, military, strike, tehran, conflict, strait

### Topic 0: AI, Datacenters, and Digital Governance / AI、数据中心与数字治理

- Conditional attention share: 0.1791
- Definition: Artificial-intelligence regulation, datacenter infrastructure, technology investment, digital risks, and related education or privacy governance.
- Naming basis: AI, datacenter, China, company, technology, power, and industry terms align with leading articles on AI regulation and infrastructure.
- Top terms: ai, datacenter, china, company, public, family, technology, center, high, school

### Topic 4: Immigration, Public Funding, and Household Economy / 移民、公共资金与家庭经济

- Conditional attention share: 0.1244
- Definition: Immigration enforcement and child impacts alongside public grants, investment, business, inflation, and household economic pressures.
- Naming basis: Immigration, ICE, child, investment, fund, grant, business, rate, and inflation terms define this broad domestic-policy cluster.
- Top terms: child, investment, immigration, ice, company, policy, fund, rate, immigrant, grant

### Topic 5: Defense Spending and Military Security / 国防开支与军事安全

- Conditional attention share: 0.1132
- Definition: Defense budgets, Pentagon management, service-member safety, military readiness, and the domestic costs of conflict.
- Naming basis: Pentagon, military, service member, defense, resolution, war costs, and security-related representative articles distinguish this topic from the broader Iran-energy topic.
- Top terms: war, iran, hegseth, pentagon, military, service_member, kill, city, conflict, resolution

### Topic 1: Infrastructure, Public Lands, and Federal Investment / 基础设施、公共土地与联邦投资

- Conditional attention share: 0.1091
- Definition: Construction, border infrastructure, public lands, grants, employment, and federal or local investment decisions.
- Naming basis: Construction, federal, city, plan, agency, grant, Big Bend, jobs, wall, and land characterize the representative articles.
- Top terms: construction, federal, city, plan, agency, job, cbp, big_bend, grant, town

## Stability

Matched-seed mean full-vocabulary cosine similarity=0.7278; mean top-15 Jaccard=0.2324.

## Interpretation

Attention shares are conditional on policy-bearing articles. Each article contributes equally regardless of length. The full three-month sample makes this a retrospective descriptive analysis, not a real-time forecasting model.
