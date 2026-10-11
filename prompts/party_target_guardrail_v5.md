# Mandatory party-target guardrail v5

Party stance is the text's evaluation **of the party, politician, or explicitly party-owned policy**. It is not the position expressed by a partisan speaker. Apply this strict target test separately for each party:

1. Complete the sentence: “The text praises/criticizes TARGET because EVIDENCE says ...”.
2. TARGET must be the named party, its politician, or a policy the article explicitly identifies as belonging to that party.
3. If TARGET is instead a bill, condition, other party, or another politician, the speaker's affiliation cannot create support for the speaker's party.
4. If the party participates in the topic but receives no evaluation, use `neutral`. If it is not meaningfully involved in the topic, use `not_expressed`.

Mandatory examples:

- “Senator Warren, a Democrat, said: ‘This law is groundbreaking.’” evaluates the law, not Warren or Democrats. Democratic stance is `neutral` if Democrats are otherwise explicitly involved, never `support` from this quote alone.
- “Democrats pilloried Trump” reports what Democrats did. It does not praise Democrats. Democratic stance is `neutral` unless separate language evaluates them.
- “A Democratic representative criticized the Republican speaker” can support `republican_stance=oppose`; it does not support Democrats.
- “Republicans would rather make it harder to vote” explicitly criticizes Republicans and can support `republican_stance=oppose`.
- “Republicans passed the bill” is neutral toward Republicans unless evaluative language targets them or an explicitly identified Republican policy.

For `support` or `oppose`, the evidence must itself show evaluation directed at the party target. When in doubt between directional stance and neutral, use `neutral`; do not infer direction from speaker identity.
