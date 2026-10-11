# Mandatory party-target guardrail v4

Before assigning either party stance, identify the **target of evaluation**, not the speaker's affiliation.

- A Democrat speaking, proposing, supporting, or criticizing something does not mean the text supports Democrats.
- A Republican speaking, proposing, supporting, or criticizing something does not mean the text supports Republicans.
- If a Democrat criticizes a Republican, this may support `republican_stance=oppose`; it does not by itself support Democrats.
- If a Republican criticizes a Democrat, this may support `democrat_stance=oppose`; it does not by itself support Republicans.
- “Democrats oppose the bill” and “Republicans passed the bill” describe party actions. If the article gives no evaluation of those parties or their identified policies, use `neutral` when the party is explicitly involved in this topic, otherwise `not_expressed`.
- Praise or criticism of a policy can count toward a party only when the article explicitly identifies that policy as the party's policy. Do not supply that link from outside knowledge.

For `support` or `oppose`, the evidence must evaluate the named party, its politician, or an explicitly linked party policy. A quotation's speaker affiliation is never sufficient evidence. Apply this target test separately to Democrats and Republicans before returning JSON.
