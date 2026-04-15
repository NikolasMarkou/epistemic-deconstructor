# Archetype Accomplice Library

Companion to `scope-interrogation.md` Mechanism M2 (Archetype Accomplice Library). Each archetype lists the **domains that typically co-drive** the target system and the **mechanism** by which each co-driver acts. Query programmatically via `scripts/scope_auditor.py enumerate --archetype <id>`; machine source is `config/archetypes.json`.

## Table of Contents

- [How to Use](#how-to-use)
- [Schema](#schema)
- [Archetypes](#archetypes)
  - [speculative_asset_market](#speculative_asset_market)
  - [regulated_infrastructure_service](#regulated_infrastructure_service)
  - [api_backed_software_service](#api_backed_software_service)
  - [supply_chain_network](#supply_chain_network)
  - [information_market](#information_market)
  - [organizational_actor](#organizational_actor)
  - [individual_persona](#individual_persona)
  - [platform_ecosystem](#platform_ecosystem)
  - [research_knowledge_producer](#research_knowledge_producer)
  - [feedback_control_system](#feedback_control_system)
- [Extending the Library](#extending-the-library)
- [Cross-References](#cross-references)

---

## How to Use

During Phase 0.7 Scope Interrogation, after classifying the target system into one or more archetypes, run:

```bash
python3 scripts/scope_auditor.py enumerate --archetype <id> --file $($SM path scope_audit.json)
```

Each returned accomplice becomes a **candidate exogenous hypothesis** to seed in `hypotheses.json` with the suggested prior. During Phase 0.7 exit gate, at least 3 such candidates must be present.

Archetypes are **orthogonal labels**, not mutually exclusive. A sovereign wealth fund managing real estate holdings is simultaneously an **organizational_actor** and a **speculative_asset_market** participant — enumerate both and take the union of accomplices (deduping by domain).

## Schema

```json
{
  "<archetype_id>": {
    "name": "human-readable name",
    "description": "one-line description",
    "examples": ["example1", "example2"],
    "accomplices": [
      {"domain": "short domain label", "mechanism": "one-line mechanism", "prior": 0.15}
    ]
  }
}
```

The `prior` field is a **tentative** prior for the hypothesis "this domain materially drives the target." Analysts should treat it as a starting point and update as Phase 0.7 flow tracing (M1) and steelman critiques (M4) produce evidence.

---

## Archetypes

### speculative_asset_market

An open-economy market where asset prices are set by a thin pool of buyers with heterogeneous motivations, openness to cross-border capital, and a store-of-value use case. Examples: real estate in tax-favorable jurisdictions, art market, collectibles, prestige passports.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| cross-border capital flows | foreign capital inflows create demand floor independent of local fundamentals | 0.25 |
| illicit finance / AML regime | high-value durable assets are laundering vehicles; price support without local demand | 0.20 |
| tax & residency regime | citizenship/residency-by-investment programs convert legal status into demand | 0.20 |
| macroprudential & monetary policy | interest-rate regime and LTV caps change effective buyer pool size | 0.25 |
| investment legislation changes | new laws open or close channels (e.g. sanctions, golden-visa revocations) | 0.20 |
| immigration / nomad flows | remote-worker relocation and non-resident demographics shift demand mix | 0.20 |
| tourism / rental-yield consumers | short-let yields reroute supply toward tourist economies | 0.15 |
| geopolitical shocks | capital flight from sanctioned/unstable regions seeking safe-haven stores | 0.15 |

### regulated_infrastructure_service

A capital-intensive service (power, water, telecom, rail) with natural-monopoly characteristics and a regulator as a primary stakeholder.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| regulatory regime | tariff caps, reliability standards, licensing regime as binding constraints | 0.30 |
| commodity input market | upstream fuel/material prices propagate via cost pass-through | 0.25 |
| climate / physical environment | weather and geophysical shocks drive demand and asset failure rates | 0.20 |
| political / tariff policy | subsidies and cross-subsidy mandates reshape unit economics | 0.20 |
| labor union dynamics | sector-wide labor agreements drive OPEX and reliability risk | 0.15 |
| adjacent infrastructure failures | cascading dependencies (grid on rail, water on power) | 0.15 |

### api_backed_software_service

A software product whose behavior depends on one or more external APIs, a hosting environment, and a community of integrators.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| upstream API provider | rate limits, schema changes, SLAs propagate as behavior changes | 0.30 |
| hosting / cloud provider | region outages, billing model changes, platform deprecations drive availability | 0.25 |
| integrator ecosystem | third-party clients expose surfaces not in the original product spec | 0.20 |
| authentication / identity provider | SSO, OAuth, CA trust roots as hidden dependencies | 0.20 |
| data-privacy regulation | GDPR/CCPA-class rules constrain storage, transit, retention | 0.15 |
| adversarial users | abuse, spam, credential-stuffing shift observed load patterns | 0.15 |

### supply_chain_network

A multi-stage system where goods, services, or materials move between heterogeneous actors with transport, inventory, and timing constraints.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| upstream commodity supplier | raw material availability and price drive downstream throughput | 0.25 |
| transport / logistics corridor | shipping routes, port congestion, freight rates as exogenous shocks | 0.25 |
| trade policy / tariffs | duties and sanctions redirect flows; rerouting creates new bottlenecks | 0.20 |
| fuel / energy market | transport costs tied to oil/gas prices | 0.20 |
| geopolitics / regional stability | chokepoint disruptions (straits, canals) stop flows | 0.15 |
| downstream demand regime | end-consumer demand shifts ripple upstream with delay | 0.20 |

### information_market

A venue where heterogeneous actors trade on asymmetric information about future events: prediction markets, betting markets, futures, narrative-driven equities.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| narrative / media cycle | coordinated attention shifts create demand not grounded in fundamentals | 0.30 |
| social-media platform dynamics | algorithmic amplification selects which narratives reach critical mass | 0.25 |
| insider / privileged information | asymmetric info produces signature trade patterns before public events | 0.20 |
| regulatory enforcement regime | SEC/CFTC/gambling-authority actions shut down or legitimize venues | 0.15 |
| market microstructure | market-maker behavior and latency regimes create non-fundamental signals | 0.20 |

### organizational_actor

A firm, agency, NGO, or coalition whose behavior emerges from internal units, incentives, and external stakeholders.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| competitor / rival org | rivals' moves drive defensive action and shape strategy | 0.25 |
| regulatory / oversight body | compliance mandates and investigations constrain action space | 0.25 |
| capital / funding source | shareholder, donor, or parent-agency demands drive priorities | 0.25 |
| labor / talent market | skill supply and wage pressures shape execution capacity | 0.20 |
| supplier / counterparty network | vendor actions cascade into the org's throughput | 0.15 |
| macro policy regime | tax, monetary, trade policy reshape unit economics | 0.15 |

### individual_persona

A human subject whose behavior emerges from traits, immediate context, and broader life systems (family, work, finances, health).

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| family / household system | domestic obligations and conflicts alter risk tolerance and timing | 0.25 |
| employer / career context | job pressure and ambition drive visible priorities | 0.25 |
| financial pressure | debts, windfalls, cashflow constraints shape decisions | 0.25 |
| health / physical state | illness, fatigue, medication alter behavior patterns | 0.20 |
| social / peer network | peer approval and reputation motivate otherwise irrational acts | 0.20 |
| ideological / identity group | group belonging dictates permissible positions | 0.15 |

### platform_ecosystem

A two- or multi-sided market where a platform operator mediates interactions between distinct user classes.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| counter-side user class | behavior on one side constrains observed dynamics on the other | 0.30 |
| platform operator policy | ranking, fees, policy enforcement reshape observed behavior | 0.30 |
| competing platform | multi-homing and switching costs drive migration patterns | 0.20 |
| regulatory antitrust regime | competition enforcement forces interop, fee caps, breakups | 0.15 |
| payment rails / identity infra | underlying payment and KYC stacks gate access | 0.15 |

### research_knowledge_producer

A lab, journal, think tank, or research network whose outputs emerge from funding, incentives, and intellectual lineages.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| funding source regime | grantor priorities pre-select which questions get asked | 0.30 |
| career incentive structure | publish-or-perish and citation metrics select for methods and topics | 0.25 |
| peer community / school of thought | dominant paradigms filter acceptable framings | 0.20 |
| tooling / data-access regime | available data and tools shape what can be measured | 0.20 |
| ethics / IRB regime | ethical review boards limit experimental designs | 0.15 |

### feedback_control_system

An engineered or natural system with a sensed state, actuator, setpoint, and control law — thermostats, autopilots, biological homeostasis, economic stabilization policies.

| Domain | Mechanism | Prior |
|--------|-----------|------:|
| sensor / measurement chain | sensor noise and drift masquerade as system dynamics | 0.25 |
| exogenous disturbance source | unmodeled external forcing drives persistent error | 0.30 |
| actuator saturation / nonlinearity | physical limits of the actuator cap achievable control | 0.20 |
| operator / human-in-loop | human overrides and schedule changes create apparent control failure | 0.20 |
| adjacent coupled system | cross-coupling between ostensibly independent loops creates emergent behavior | 0.15 |

---

## Extending the Library

To add a new archetype:

1. Edit `config/archetypes.json`. Add a new top-level key following the schema.
2. List 4-8 accomplices. Keep mechanisms one-line and specific.
3. Set tentative priors in [0.10, 0.35]. Avoid extremes — these are starting points.
4. Document the archetype in this file with the same structure (description + table).
5. Run `python3 scripts/scope_auditor.py enumerate --archetype <new_id>` to verify it loads.

**Design rules for new archetypes:**

- **Orthogonal**, not hierarchical. Don't nest archetypes; allow systems to match multiple.
- **Generalizable**. If an archetype only applies to one specific instance ("Cyprus real estate"), it is too narrow — generalize upward to "open-economy speculative asset market."
- **Mechanism, not just label**. Each accomplice needs a one-line mechanism — "legislation" alone is not an accomplice; "tax residency law changes the effective buyer pool" is.
- **Empirically grounded**. Base new archetypes on real observed co-driver patterns, not on speculation.

---

## Cross-References

- Protocol entry: `scope-interrogation.md` (M2 mechanism, Phase 0.7 procedure)
- Agent: `agents/scope-auditor.md` runs this library at Phase 0.7
- Tool: `scripts/scope_auditor.py` loads and queries `config/archetypes.json`
- Simulation archetypes (orthogonal classification): `simulation-guide.md`
- Cognitive traps that scope interrogation addresses: `cognitive-traps.md` (Framing, Streetlight, OVB, Premature Closure)
