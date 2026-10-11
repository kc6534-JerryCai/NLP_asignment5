# Guardian US midterm-election LDA analysis

This is a retrospective descriptive analysis of the full three-month Guardian sample. It is not a no-look-ahead real-time forecasting model and does not measure US voter preferences.

## Sample

- Input articles: 1,216
- Contextually screened relevant articles before modeling filters: 641
- Uncertain articles reserved for sensitivity analysis: 424
- Final LDA documents: 638
- Dictionary terms: 6,346

## Model choice

Selected K=10. Its c_v coherence is 0.4282. Selection must combine coherence, keyword overlap, representative articles and interpretability; it is not based on election probabilities or market returns.

## Interpreted topics

### Topic 7: Primaries and Candidate Competition / 初选与候选人竞争

- Overall attention share: 0.2016
- Category: campaign_process_or_election_strategy
- Definition: Party primaries, candidate recruitment, ideological competition, and strategies for winning congressional races.
- Naming basis: Democrats, campaign, voter, candidate, primary, and race terms align with the representative campaign articles.
- Top terms: state, democrats, campaign, voter, win, candidate, party, republican, vote, michigan

### Topic 1: Iran War and Foreign Policy / 伊朗战争与外交政策

- Overall attention share: 0.1268
- Category: policy_or_social_issue
- Definition: US military action, diplomacy, and political consequences surrounding Iran and the wider Middle East.
- Naming basis: Iran, war, attack, military, strike, and diplomacy terms align with the highest-probability articles.
- Top terms: iran, war, president, attack, iranian, military, strike, deal, world, israel

### Topic 3: Election Administration and Voting Rights / 选举管理与投票权

- Overall attention share: 0.1057
- Category: campaign_process_or_election_strategy
- Definition: Federal and state control of election rules, ballots, monitoring, and voting access.
- Naming basis: Election, voter, ballot, voting, court, and rule terms match articles on federal election powers and mail voting.
- Top terms: election, state, president, voter, administration, vote, white_house, federal, trump_administration, ballot

### Topic 9: Economy, Tariffs, and Cost of Living / 经济、关税与生活成本

- Overall attention share: 0.1052
- Category: policy_or_social_issue
- Definition: Economic performance, trade and tariffs, inflation, prices, and affordability in the midterm context.
- Naming basis: Tariff, economy, price, inflation, trade, rate, and cost terms align with the leading articles.
- Top terms: tariff, president, company, government, economy, price, canada, high, rate, datacenter

### Topic 8: Congress, Legislation, and Public Spending / 国会、立法与公共支出

- Overall attention share: 0.0986
- Category: policy_or_social_issue
- Definition: Congressional voting, funding bills, foreign aid, and legislative conflict between the parties.
- Naming basis: House, Senate, bill, pass, funding, and vote terms characterize the representative articles.
- Top terms: vote, democrats, house, senate, republicans, bill, republican, president, support, party

### Topic 0: Senate Succession and Political Movements / 参议院继任与政治运动

- Overall attention share: 0.0938
- Category: campaign_process_or_election_strategy
- Definition: Candidate succession, political movements, and their effects on party competition in Senate and congressional races.
- Naming basis: Graham succession coverage and articles on progressive movements and candidates dominate the representative set.
- Top terms: graham, state, win, senate, south_carolina, republican, voter, president, run, campaign

### Topic 4: Political Culture and Executive Conduct / 政治文化与行政行为

- Overall attention share: 0.0844
- Category: mixed_or_unclear
- Definition: Public controversy over presidential rhetoric, authoritarian conduct, gender, media symbolism, and political spectacle.
- Naming basis: Representative articles center cultural backlash and executive conduct, but the topic is substantively broad.
- Top terms: president, state, post, hegseth, woman, white_house, military, republican, campaign, texas

### Topic 6: Public Health and Political Accountability / 公共卫生与政治问责

- Overall attention share: 0.0819
- Category: mixed_or_unclear
- Definition: Public-health controversies and hearings linked to political careers, accountability, and campaigning.
- Naming basis: Fauci hearings and health-policy articles dominate, alongside political biographies and accountability stories.
- Top terms: state, fauci, health, public, political, mcconnell, republican, issue, office, know

### Topic 2: Justice Department and Executive Oversight / 司法部与行政监督

- Overall attention share: 0.0777
- Category: mixed_or_unclear
- Definition: Appointments, hearings, investigations, and institutional checks involving the Justice Department and executive branch.
- Naming basis: Todd Blanche hearings and Justice Department oversight lead the representative articles, with some adjacent institutional disputes.
- Top terms: president, blanche, state, vote, fund, election, justice_department, attorney_general, white_house, donald_trump

### Topic 5: White House Personnel and Information Authority / 白宫人事与信息权力

- Overall attention share: 0.0244
- Category: mixed_or_unclear
- Definition: Senior appointments, intelligence leadership, press operations, and disputes about official truth and legitimacy.
- Naming basis: Intelligence and press appointments, White House personnel, and election-legitimacy claims lead the representative set.
- Top terms: president, election, clayton, white_house, intelligence, leavitt, china, american, claim, pulte

## Stability

Two additional seeds were matched to the base topics with Hungarian assignment. Mean full-vocabulary cosine similarity was 0.7909; mean top-15 Jaccard was 0.3638.

## Attention definition

Overall attention is the mean full topic probability across effective articles. Daily attention is the mean full topic probability across articles published on each New York local date. Each article has equal weight regardless of length. Dates with no effective articles remain missing. The plotted 7-day series is the unweighted rolling mean of available calendar-day daily shares and is for display only.
