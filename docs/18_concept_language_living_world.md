# Concept: an ever-evolving task ladder, a social system in which language can emerge, and a living world

Date: 2026-10-13. Status: concept for decision, nothing built. It starts from what DNA-3 measured (docs/17), checks
every part of the current machine against the goal, and proposes DNA-4 in stages, each with an acceptance test.

## 1. Where we stand

DNA-3 built six days of a world on top of the DNA-2 replicator (docs/17): light and energy, a 2D grid with a
firmament, land, sea and plants, seasons, predators, movers, parasites, tasks, and signals. Its long worlds showed:

- computation evolves from plants that had none, within 30,000 ticks (O6);
- with a fixed ladder of four tasks the world reaches a plateau (O7, D7);
- a longer ladder with depletable pools keeps new adaptive sequences coming at about 1,400 per census (D8), but most of
  what it climbed was a loophole (answering only easy inputs);
- with that loophole closed (`--answer-all`, D9) organisms evolved **x+1 with a real carry**; there is still **no x-1
  with a borrow** and **no signalling channel**: signals carry 0.14 bits about the sender's input, listeners do no
  better than chance on the social task, and the social rewards that do occur are consistent with chance streaks.

The two gaps have the same root: the machine and the rewards make honest computation on two values expensive and
make luck cheap.

## 2. System check

| element | now (DNA-3, D9) | consequence for tasks and language | suggestion |
|---|---|---|---|
| genome cell | 2 bits (A, C, G, T) | the dense alphabet and the degenerate code are what made replication common (docs/15, 16) | **keep** |
| codon | 3 cells = 6 bits, 64 codons | | keep |
| genetic code | **full**: 64 codons on 31 operations + STOP; 20 operations have 2 codons, 6 have 1 (WAIT, SENSE, SIGNAL, LISTEN, IN, OUT, REWIND) | every new instruction must take codons from the copy machinery, whose synonyms made origins about 40 times faster (docs/16); the newest instructions (signals, I/O) have only one codon each, so mutation reaches them rarely | add an **extension prefix** instead of taking more codons (section 3.1) |
| accumulator A | 2 bits | one base-4 digit; LISTEN overwrites it, so a received value replaces the own one | keep as a digit register; add a **carry flag** and a second register |
| operations on values | INC and FLIP on the cell under the head, LDA, STA, skips on the cell | **no ADD, no XOR, no DEC**: combining two values (b xor b', a borrow) needs long branching programs | two-operand **ALU on digits** (XOR, ADD with carry, DEC, compare) through the extension space |
| memory | the tape only; writes in the own slot can damage the gene | a listener cannot hold a received symbol and its own input at once without risking its gene | a small **scratchpad** in the processor (16 cells of 2 bits), outside the genome |
| input | 4 bits in the slot's last two cells (G-2, G-1) | enough for 4-bit arithmetic | widen with the ladder (8 bits = 4 cells) |
| signal | 2 bits per tick, one register, read by LISTEN in a direction chosen by the codon | one symbol per tick and no memory of the previous ones: no words, no sequences | keep 2-bit symbols as "phonemes"; add a **receive queue** of the last 8 symbols per processor |
| social task | low cell = b xor b' of the listened neighbour, streak 7, rank 3 | needs a computation that costs about as much as the hardest ladder tasks, pays the listener only, and is matched by chance often enough to be paid billions of times | replace with **grounded referential situations** in which the world pays both partners (section 3.3) |
| task ladder | 19 fixed tasks, ranked by the shortest terminating program, pools per 32 x 32 region, anti-guessing streaks, answer-all | finite (climbed or bypassed), and loopholes were found by hand twice | **generative ladder** with automatic admission screening (section 3.2) |
| slot | 256 cells = 85 codons, genes usually 13 to 20 codons | about 190 cells of padding per slot | keep; use the padding for read-through and evolvability, not for memory |
| budget | at most 16 instructions per tick, paid from energy | talking and computing compete with copying and eating: the right pressure | keep |
| world | 3.0 x 10^9 cells, 6 M organisms, about 20 ticks/s on the RTX 4090 | about 30 generations per day of run time at that size | develop at 10^7 to 10^8 cells with replicates and controls; the big world for confirmation |
| verification | reference and engine identical on every tick | | **keep as a hard rule**, including for every learned component |

## 3. The concept: DNA-4

Five layers, built in order. Each one is useful on its own and is tested before the next one starts.

### 3.1 Layer A: the protein machine (registers, ALU, scratchpad)

Biology keeps DNA simple and puts the richness in proteins. Do the same.

- **Extension prefix.** One codon becomes `EXT`, and the codon after it selects one of 64 extended operations,
  like x86's 0x0F prefix. Candidate: TGA, which is a STOP in the standard code but codes for Trp in vertebrate
  mitochondria, so the reassignment has a natural precedent. The base code and its synonyms stay as they are, and two-codon
  instructions are rarer under mutation, so the new operations are found when they pay.
- **Extended operations, first set**: `XORC` (cell ^= A), `ADDC` (cell += A + carry, sets carry), `DEC`, `CMP`
  (skip if cell == A), `CLC`/`SKC` (clear and test carry), `LDS i`/`STS i` (scratchpad cell i to and from A),
  `PUSHQ`/`POPQ` (send a symbol, read the oldest received symbol). With these, b xor b' is `LISTEN XORC`, and x-1 with
  a borrow is a short digit-serial loop, so the ladder can rank tasks by what they compute, not by how awkward the
  machine is.
- **Acceptance test.** The exhaustive sweep of docs/11 and 15 rerun with the extension: replicator density must not
  fall by more than a factor of 2 (the extension must not break life). The ladder's tasks are re-ranked by shortest
  program.

### 3.2 Layer B: a generative task ladder that never runs out

- **Tasks are generated, not listed.** A task is a function of the input built by composing primitives (the docs/07
  vocabulary plus the ALU operations): every new task is `compose(solved task, primitive)` or a wider version of a
  solved one (4-bit to 8-bit input). Stepping stones are built in.
- **Admission screening.** Before a task enters the world, a sweep checks it automatically: every program up to
  length k on the organisms' machine, including guessing, constant, echo, abstaining and re-drawing strategies. A task
  enters only if no strategy shorter than its honest program earns more than (1/4)^streak of its reward. The two
  loopholes DNA-3 found by hand (guessing, abstaining) become standing tests.
- **Prices move.** Rewards fall with the share of a region that already solves a task (the pools of D8, kept), and the
  first lineage to solve a new task in a region collects a full pool (a novelty bounty). A task solved by most of the
  world for long enough is retired and spawns harder children.
- **Organisms become the environment.** The richest source of endless challenges is other organisms (autocurricula,
  Leibo et al. 2019; POET, Wang et al. 2019): predators against prey, parasites against hosts, and in layer C speakers
  against deceivers. The generated ladder is the floor; coevolution supplies the open end.
- **Acceptance test**: the day-7 criteria of docs/17 section 9 hold for at least four times the age at which the last
  externally generated task appeared.

### 3.3 Layer C: a social system grounded in the world

Language evolves when there is something worth saying, one partner knows it and the other does not, and both gain when
the message gets through. DNA-3's social task had none of these properties, so this layer replaces it.

**Situations the world creates** (no answer matching, so nothing is paid by coincidence):

1. **Fruit.** Rich energy patches appear for a limited time at random places. Only organisms within a short sensing
   radius see one (`SENSEF` gives direction 0 to 7 and distance 0 to 3); fruit lasts long enough to reach from farther
   away if one knows where it is. A speaker who sees fruit can signal; listeners can move toward it.
2. **Alarms.** Predators already exist (77 % of D9's organisms eat). A signal that makes kin hide (WAIT) or flee before
   a predator arrives saves them. With three predator kinds (land, sea, air) the vervet monkey's three alarm calls are
   the model (Seyfarth, Cheney and Marler 1980).
3. **Joint tasks.** Some food can only be taken by two organisms acting in the same tick, so coordination pays.

**Payoffs are shared.** A listener records whom it listened to; when it eats fruit within k ticks, the speaker gets a
finder's share. Children are placed next to their parents, so neighbours are often kin. Floreano et al. (2007) found
that communication evolves reliably in robot groups under kin and group selection and turns deceptive under individual
selection; this world will have both, and the arms race between honest and deceptive signals is itself an open-ended
driver.

**The meaning space is larger than the alphabet.** 8 directions x 4 distances x 4 food kinds = 128 meanings against
4 symbols per tick: a message must be a sequence, so there is pressure for structure (one symbol for direction, one
for distance, ...), the precondition of compositionality.

**Acceptance tests, measured against controls.** In paired control worlds LISTEN returns a random neighbour's symbol
(noise) or nothing (deaf). Communication is shown only if (i) fruit found per organism is higher than in both controls,
(ii) the mutual information between meaning and message, and between message and the listener's move, is above the
shuffled baseline, and (iii) removing the channel mid-run (an ablation fork) costs energy.

### 3.4 Layer D: culture, so that a language can be learned and change

Genetic signals give fixed codes, like alarm calls or a bee dance. A language in the human sense is learned, passed
on and changing. That needs a lexicon outside the genome:

- each processor carries a **lexicon** of 16 entries (meaning class to symbol sequence, and back), part of its
  state, hashed and checkpointed like everything else;
- `SAY m` emits the sequence for meaning m; `HEAR` decodes the received queue into a meaning;
- **learning**: after a successful outcome (fruit reached, alarm survived) the entries used are reinforced; after a
  failure they mutate;
- **transmission**: a child copies its parent's lexicon at DIVIDE (vertical); an organism copies a successful
  neighbour's entry with some probability (horizontal, conformity);
- **bottleneck**: the lexicon holds fewer entries than there are meanings, so meanings must share structure. In
  iterated learning a transmission bottleneck makes languages compositional (Kirby, Cornish and Smith 2008), and the
  naming games of Steels show how a population agrees on a shared vocabulary without any central control.

This is deterministic and verifiable (the update rule is arithmetic on state), so reference and engine stay identical.

**What would count as "a real language" here**, measurable:

| property | measure |
|---|---|
| referential | mutual information meaning to message above controls, and ablation cost |
| conventional | different regions settle on different words for the same meaning (dialects), especially across the firmament |
| compositional | topographic similarity: similar meanings get similar messages, beyond chance |
| productive | messages for meaning combinations never rewarded before are understood |
| cultural | the lexicon changes faster than the genes, and conventions survive the death of their inventors |

### 3.5 Layer E: minds (later)

The interface of docs/17 section 8 stays the plan: a learned controller (enderPeer/matrix's factored Q learner) that
chooses among a gene's own subroutines, with parameters inherited or learned (Baldwin and Lamarck variants). It comes
after layers A to D, when the world has things worth learning about.

## 4. The living world: what keeps it moving

The world needs reasons to move, meet and talk that do not run out:

- **day, night and seasons** move the fruit and the light, so food is never where it was (days 1 and 4);
- **two hemispheres** behind the firmament with opposite seasons, so migration by flying pays twice a year (days 2,
  4, 5);
- **predators and parasites** coevolve with prey and hosts, since DNA-3 has both;
- **fruit and alarms** make information valuable and perishable;
- **culture** changes conventions faster than genes, so neighbours keep having to keep up.

## 5. Plan and cost

| stage | what | test | where and roughly how long (estimates) |
|---|---|---|---|
| A | extension prefix, ALU, scratchpad, carry; reference and engine | replicator density within 2x; ladder re-ranked; 30+ configurations identical | a day of work; sweeps minutes to an hour on the 4090 |
| B | generative ladder and admission screening | loophole screen catches the two known loopholes; day-7 criteria | a day; runs 10^8 cells on the 3060s |
| C | fruit, alarms, joint food, shared payoffs, receive queue | communication beats noise and deaf controls; mutual information; ablation | two days; 3 replicates x 3 controls at 10^8 cells |
| D | lexicon, learning, transmission | dialects, compositionality, change rate | two days; same scale |
| E | learned controllers | Baldwin effect | a separate project |

The big world (3 x 10^9 cells) is used once per stage, for confirmation and pictures, not for discovery: at 20 ticks
per second it gives about 30 generations per day, while a 10^8-cell world gives many more per day and can be
replicated.

## 6. Open decisions for the user

1. **Extension prefix**: take TGA (one of three STOP codons, mitochondrial precedent), REWIND's TGG, or a codon of
   another operation?
2. **ALU scope**: digit-serial base-4 arithmetic only (XOR, ADD with carry, DEC, compare), or also multiply and shifts?
3. **Scratchpad and lexicon outside the genome**: acceptable, or should all memory live on the tape (harder, but
   "everything is DNA")?
4. **Fruit**: lifetime, size and sensing radius; energy worth relative to plants (it must be worth travelling for).
5. **Shared payoffs**: finder's share for speakers (common interest) from the start, or only kin selection first, to
   see whether communication evolves without it?
6. **Deception**: allowed from the start (an arms race), or a first phase in which signals cannot be false?
7. **Learning**: genetic codes only first (fixed signals), then the lexicon; or the lexicon from the start?
8. **What counts as success**: the five properties of section 3.4 as stated, or a different bar?
9. **D9**: keep it running as the long-run control, or stop it to free the 4090?
10. **Big-world use**: confirmation only (as proposed), or a long showcase run per stage?

## 7. References

- Bedau and Packard (1992): evolutionary activity statistics.
- Floreano, Mitri, Magnenat and Keller (2007): evolutionary conditions for the emergence of communication in robots. Current Biology 17.
- Kirby, Cornish and Smith (2008): cumulative cultural evolution in the laboratory. PNAS 105.
- Lazaridou, Peysakhovich and Baroni (2017): multi-agent cooperation and the emergence of (natural) language. ICLR.
- Leibo, Hughes, Lanctot and Graepel (2019): autocurricula and the emergence of innovation from social interaction.
- Lenski, Ofria, Pennock and Adami (2003): the evolutionary origin of complex features. Nature 423.
- Lewis (1969): Convention (signalling games); Skyrms (2010): Signals.
- Mordatch and Abbeel (2018): emergence of grounded compositional language in multi-agent populations. AAAI.
- Seyfarth, Cheney and Marler (1980): vervet monkey alarm calls. Science 210.
- Steels (1995, 2003): naming games and the Talking Heads experiment.
- Wang, Lehman, Clune and Stanley (2019): POET, paired open-ended trailblazer.
