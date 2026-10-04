"""Synthetic long-lived assistant users with labeled context requirements.

Each generated user has roughly six months of history with an assistant:

- a static profile (name, role, experience, languages, response style, personal facts)
- a database preference that is sometimes superseded by a later switch
- two or three projects, each with an introduction, technical decisions (some later revised),
  and a sequence of status updates
- many distractor sessions: generic technical Q&A on overlapping topics and unrelated small talk

Queries are generated from the same facts, so every query has exact ground truth: the memory ids
needed to answer it, the stale ids that would produce an outdated answer, and keywords a correct
answer must contain. The phrasing of queries deliberately differs from the phrasing of the
memories they need, and distractors share vocabulary with the facts, so no single retrieval
signal is sufficient.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from contextio.bench.dataset import BenchQuery, BenchUser, Dataset
from contextio.memory import Memory, MemoryKind, heuristic_importance

HORIZON_DAYS = 180.0

Event = tuple[float, int, Callable[[float], None]]
"""(day, priority, emit): emitted in day order to build a chronological history."""

NAMES = (
    "Priya Aiden Mei Lucas Amara Diego Hana Noah Zara Mateo Ingrid Kofi Lena Ravi Sofia Tomas "
    "Yuki Omar Elena Jonah Aisha Felix Nadia Leo Imani Sven Carmen Arjun Maya Theo"
).split()
ROLES = (
    "backend engineer",
    "data scientist",
    "frontend developer",
    "machine learning engineer",
    "site reliability engineer",
    "mobile developer",
    "platform engineer",
    "computer science student",
)
LEVELS = ("beginner", "intermediate", "senior", "staff-level")
LANGUAGE_SETS = (
    ("Python", "SQL"),
    ("Go", "Rust"),
    ("TypeScript", "Node.js"),
    ("Java", "Kotlin"),
    ("C++", "Python"),
    ("Swift", "Objective-C"),
    ("Rust", "C"),
    ("Scala", "Python"),
    ("Ruby", "JavaScript"),
    ("C#", "F#"),
)
STYLES = (
    "concise answers with code examples",
    "detailed step-by-step explanations",
    "bullet-point summaries",
    "answers that lead with the trade-offs",
    "an analogy first, then the technical detail",
)
CITIES = ("Lisbon", "Toronto", "Bangalore", "Berlin", "Austin", "Nairobi", "Seoul", "Melbourne")
HOBBIES = ("rock climbing", "film photography", "woodworking", "board games", "cycling", "piano")
PETS = ("a beagle named Pixel", "two cats", "a parrot", "a rescue greyhound", "no pets")

DATABASES = {
    "PostgreSQL": "the JSONB support",
    "MySQL": "the replication tooling",
    "MongoDB": "the flexible schemas",
    "SQLite": "the zero-ops setup",
    "DynamoDB": "the managed scaling",
    "CockroachDB": "the multi-region support",
}
DB_INITIAL = (
    "I mostly use {db} for my side projects, it's what I know best.",
    "For what it's worth, I usually reach for {db} whenever I need a database.",
    "My go-to database has been {db} for a few years now.",
)
DB_SWITCH = (
    "Quick note: I've switched to {db} for everything now, {reason} won me over.",
    "I've moved over to {db} for all new work; {reason} made the difference.",
    "Heads up, I no longer use {old}. I prefer {db} these days because of {reason}.",
)
DB_QUERIES = (
    "What database should I use for the new service I'm building?",
    "I'm spinning up a new app. Which database would you suggest for me?",
    "Pick a datastore for my next project, keeping my preferences in mind.",
)

PROJECTS = (
    "Ledgerly Trailmix Beacon Orbit Harvest Quill Atlas Pulse Nimbus Sprout Relay Mosaic Lumen "
    "Cobalt Fathom"
).split()
DOMAINS = (
    "a budgeting app for freelancers",
    "a hiking trail recommendation service",
    "an internal incident dashboard",
    "a recipe sharing platform",
    "a real-time chat backend",
    "a document search engine",
    "a fitness tracking API",
    "a URL shortener with analytics",
    "a CI build-cache service",
    "an IoT sensor ingestion pipeline",
)
ASPECTS = {
    "caching layer": ("caching", ("Redis", "Memcached", "an in-process LRU cache", "Varnish")),
    "message queue": ("queues", ("Kafka", "RabbitMQ", "Amazon SQS", "NATS")),
    "deployment platform": (
        "deployment",
        ("Kubernetes", "AWS Lambda", "Fly.io", "a single VM with Docker Compose"),
    ),
    "authentication": ("auth", ("OAuth with Google", "magic email links", "JWT sessions", "Auth0")),
    "API style": ("api", ("REST", "GraphQL", "gRPC")),
    "frontend framework": ("frontend", ("React", "Svelte", "server-rendered templates", "Vue")),
    "testing strategy": (
        "testing",
        ("property-based tests", "contract tests", "end-to-end Playwright tests"),
    ),
}
REASONS = (
    "it keeps operations simple",
    "everyone on the team already knows it",
    "it was the cheapest option at our scale",
    "latency mattered more than anything else",
    "it scales without much tuning",
    "the documentation is excellent",
)
DECISION_TEMPLATES = (
    "For {project}, we decided to go with {choice} for the {aspect} because {reason}.",
    "Okay, settled: {project} will use {choice} for the {aspect}. Mainly because {reason}.",
    "After a lot of back and forth we picked {choice} as the {aspect} for {project}, "
    "since {reason}.",
    "{choice} it is for the {aspect} on {project}; {reason}.",
)
REVISION_TEMPLATES = (
    "Change of plans on {project}: we switched the {aspect} from {old} to {choice} "
    "because {reason}.",
    "We decided to drop {old} on {project} and moved the {aspect} over to {choice}; {reason}.",
)
DELIBERATION_TEMPLATES = (
    "I'm torn between {a} and {b} for the {aspect} in {project}. Thoughts?",
    "What are the trade-offs of {a} versus {b} for a {aspect}? Asking for {project}.",
)
DECISION_QUERIES = (
    "Remind me what we settled on for the {aspect} in {project}?",
    "What did we decide to use for the {aspect} on {project}?",
    "For {project}, which {aspect} option did we end up choosing?",
)
INTRO_TEMPLATES = (
    "I'm starting a new project called {project}: {domain}. Planning to build it with {stack}.",
    "New side project alert. {project} is {domain}, and I'm writing it in {stack}.",
    "Been hacking on {project} lately, {domain} built on {stack}.",
)
STATUSES = (
    ("the schema is finalized and I'm now wiring up the HTTP handlers", "handlers"),
    ("the core API works end to end and I'm writing integration tests", "integration tests"),
    ("we shipped a private beta to ten early users", "beta"),
    ("I'm blocked on slow queries in the reporting page", "slow queries"),
    ("auth is finished and billing is next on the list", "billing"),
    ("I'm migrating the background jobs to a proper scheduler", "background jobs"),
)
STATUS_TEMPLATES = (
    "Quick update on {project}: {status}.",
    "Status on {project}: {status}.",
    "{project} update, {status}.",
    "Progress on {project}: {status}.",
)
CONTINUATION_QUERIES = (
    "Let's pick up where we left off on {project}.",
    "Can you help me plan the next steps for {project}?",
    "Where do things stand with {project}, and what should I tackle next?",
)
BACKGROUND_QUERIES = (
    "Explain how async/await works, in terms that fit my background.",
    "Walk me through how garbage collection works, pitched at my level and using languages I know.",
    "Explain dependency injection with an example in a language I actually use.",
)
STYLE_QUERIES = (
    "Give me a quick rundown of consistent hashing.",
    "How do bloom filters work?",
    "What's the difference between a process and a thread?",
)
ASIDES = (
    "I usually go for a run before work.",
    "I prefer green tea over coffee these days.",
    "We decided to repaint the kitchen this weekend.",
    "My team is moving to a four-day week.",
    "I've switched to Obsidian for my notes.",
    "I mostly listen to lo-fi beats while coding.",
    "I'm thinking about learning Spanish.",
    "Update: the apartment move is finally done.",
    "We picked a new sprint cadence at work, two weeks instead of one.",
    "I always forget to drink water when I'm focused.",
    "My team is hiring two new engineers.",
    "I never check email before lunch anymore.",
    "We agreed to do a team offsite in the spring.",
    "I've moved over to a standing desk.",
    "I'm working on a talk for a local meetup.",
    "Note that I'm off next Friday.",
    "Status: still recovering from a cold.",
    "I prefer mechanical keyboards, the louder the better.",
    "We chose a new office plant, it's a fiddle leaf fig.",
    "My company switched to a new expense tool and everyone hates it.",
)
TECH_ASIDES = (
    "A friend keeps telling me to try {db}, but I haven't had the time.",
    "Someone at work said {db} handles analytics workloads better, not sure that's true.",
    "Saw a conference talk about {db} internals, pretty interesting.",
)
GREETINGS = ("Hey!", "Morning.", "Back again.", "Hope you're well.", "Hi there.")
SIGNOFFS = (
    "Anyway, that's where things are.",
    "Thanks again for the help last time.",
    "Long week, but getting there.",
)


@dataclass(frozen=True)
class Topic:
    questions: tuple[str, ...]
    sentences: tuple[str, ...]


TOPICS: dict[str, Topic] = {
    "caching": Topic(
        (
            "Can you explain common caching strategies?",
            "When does adding a cache actually make things worse?",
        ),
        (
            "Caching trades memory for latency by keeping hot data close to where it is used.",
            "Cache invalidation is famously hard because stale entries can silently serve "
            "wrong data.",
            "A common pattern is cache-aside, where the application checks the cache before "
            "the database.",
            "Time-to-live values bound staleness but can cause thundering herds when many keys "
            "expire together.",
            "Distributed caches such as Redis or Memcached add network hops, so in-process "
            "caches often win for small, read-heavy datasets.",
            "Measuring hit ratio is the quickest way to tell whether a cache is pulling its "
            "weight.",
        ),
    ),
    "queues": Topic(
        (
            "How do message queues help with scaling?",
            "What's the difference between Kafka and RabbitMQ in practice?",
        ),
        (
            "Message queues decouple producers from consumers so each side can scale "
            "independently.",
            "At-least-once delivery means consumers must be idempotent to handle duplicates "
            "safely.",
            "Partitioning a topic increases throughput but only preserves ordering within a "
            "partition.",
            "Dead-letter queues capture messages that repeatedly fail so they can be inspected "
            "later.",
            "If consumers fall behind, queue depth is the first metric to watch.",
            "Log-based brokers like Kafka retain messages, which lets new consumers replay "
            "history.",
        ),
    ),
    "databases": Topic(
        (
            "How should I think about choosing a database?",
            "When is a document database a better fit than a relational one?",
        ),
        (
            "Relational databases shine when data has clear relationships and needs "
            "transactional guarantees.",
            "Document stores make schema changes cheap but push consistency rules into "
            "application code.",
            "Indexes speed up reads at the cost of slower writes and extra storage.",
            "Connection pooling is often the first fix when a database falls over under load.",
            "Read replicas help with read-heavy traffic but introduce replication lag.",
            "Choosing a database is usually more about operational familiarity than raw "
            "benchmarks, whether that is PostgreSQL, MySQL, or MongoDB.",
        ),
    ),
    "deployment": Topic(
        (
            "What are the trade-offs between containers and serverless?",
            "How do teams usually do zero-downtime deploys?",
        ),
        (
            "Containers package an application with its dependencies so it runs the same "
            "everywhere.",
            "Orchestrators such as Kubernetes handle scheduling, restarts, and rolling updates, "
            "at the cost of complexity.",
            "Serverless platforms remove server management but introduce cold starts and "
            "execution limits.",
            "Blue-green deployments keep the previous version running so rollbacks are instant.",
            "Infrastructure as code makes environments reproducible and reviewable.",
            "For small services, a single well-monitored virtual machine is often enough.",
        ),
    ),
    "auth": Topic(
        (
            "What's the difference between authentication and authorization?",
            "Are JWTs a good idea for sessions?",
        ),
        (
            "Authentication answers who a user is, while authorization decides what they may do.",
            "Short-lived access tokens paired with refresh tokens limit the damage of a leaked "
            "token.",
            "Delegating login to an identity provider avoids storing passwords yourself.",
            "Session cookies should be marked HttpOnly and Secure to reduce theft via scripts.",
            "Rate limiting login attempts is a cheap defense against credential stuffing.",
        ),
    ),
    "api": Topic(
        (
            "REST or GraphQL, how do people decide?",
            "Any advice on designing a public API?",
        ),
        (
            "REST maps resources to URLs and leans on HTTP semantics for caching and errors.",
            "GraphQL lets clients ask for exactly the fields they need, which helps mobile apps.",
            "gRPC uses protocol buffers and HTTP/2, which suits low-latency service-to-service "
            "calls.",
            "Versioning an API early saves painful migrations once external clients depend on it.",
            "Cursor-based pagination is more robust than offsets when data changes underneath.",
        ),
    ),
    "frontend": Topic(
        (
            "Is server-side rendering worth it?",
            "How do I keep frontend state from getting messy?",
        ),
        (
            "Component-based frameworks make it easier to reuse UI pieces across pages.",
            "Server-side rendering improves first paint and helps search engines index content.",
            "Keeping global state small avoids a whole class of hard-to-trace UI bugs.",
            "Bundle size directly affects load time on slow mobile connections.",
            "Frameworks like React, Vue, and Svelte differ more in ergonomics than in capability.",
        ),
    ),
    "testing": Topic(
        (
            "How much should I invest in end-to-end tests?",
            "What is property-based testing good for?",
        ),
        (
            "Unit tests are fast and precise but can miss problems that only appear when parts "
            "interact.",
            "Property-based testing generates many inputs automatically and often finds edge "
            "cases humans miss.",
            "Contract tests catch breaking changes between services without spinning up the "
            "whole system.",
            "End-to-end tests give the most confidence but are slow and prone to flakiness.",
            "Mutation testing measures whether a test suite actually detects injected bugs.",
        ),
    ),
    "baking": Topic(
        ("Any tips for baking sourdough?", "Why is my bread so dense?"),
        (
            "Sourdough rises more predictably when the starter is fed on a consistent schedule.",
            "A longer cold proof in the fridge deepens flavor and makes scoring easier.",
            "Hydration around seventy percent is a forgiving place to start for an open crumb.",
            "Preheating a Dutch oven traps steam and gives the crust a better rise.",
        ),
    ),
    "running": Topic(
        ("How should I train for a marathon?", "How do I avoid hitting the wall on long runs?"),
        (
            "Most marathon plans build mileage gradually, adding no more than ten percent per "
            "week.",
            "Easy runs should feel conversational; the hard work happens in a few targeted "
            "sessions.",
            "Fueling during long runs prevents the dreaded wall around the thirty kilometer mark.",
            "Tapering for two to three weeks before race day lets the body absorb the training.",
        ),
    ),
    "travel": Topic(
        ("Planning a trip to Japan, any advice?", "Is a rail pass worth it in Japan?"),
        (
            "Rail passes can pay off quickly if you plan several long-distance trips.",
            "Kyoto is quieter early in the morning, before tour groups arrive at the temples.",
            "Convenience stores in Japan are genuinely good for quick, cheap meals.",
            "Carrying some cash is still useful in smaller towns and family-run restaurants.",
        ),
    ),
    "plants": Topic(
        ("My houseplants keep dying, what am I doing wrong?", "How often should I repot plants?"),
        (
            "Most houseplants die from overwatering rather than neglect.",
            "Bright, indirect light suits the majority of common tropical houseplants.",
            "Repotting is best done in spring when plants are actively growing.",
            "Yellowing lower leaves often signal that the soil is staying wet for too long.",
        ),
    ),
    "chess": Topic(
        ("How do I get better at chess?", "Which chess opening should a beginner learn?"),
        (
            "Controlling the center early gives your pieces more room to maneuver.",
            "Developing knights before bishops is a common guideline in many openings.",
            "Castling early keeps the king safe and connects the rooks.",
            "Studying endgames often improves results faster than memorizing opening lines.",
        ),
    ),
}
DISTRACTOR_TOPICS = tuple(TOPICS)
ACKS = ("Good question.", "Happy to help.", "Sure.", "Here's how I'd think about it.", "")


class _UserBuilder:
    def __init__(self, user_id: str, rng: random.Random) -> None:
        self.user_id = user_id
        self.rng = rng
        self.memories: list[Memory] = []
        self.queries: list[BenchQuery] = []
        self._turns_at: dict[float, int] = {}
        self._pending: list[Callable[[], None]] = []

    def _memory(self, text: str, day: float, kind: MemoryKind = MemoryKind.EPISODIC) -> str:
        offset = self._turns_at.get(day, 0)
        self._turns_at[day] = offset + 1
        memory = Memory(
            id=f"{self.user_id}-m{len(self.memories):04d}",
            text=text,
            kind=kind,
            timestamp=0.0 if kind is MemoryKind.STATIC else round(day + offset * 0.001, 3),
            importance=heuristic_importance(text),
        )
        self.memories.append(memory)
        return memory.id

    def _static(self, text: str) -> str:
        return self._memory(text, 0.0, MemoryKind.STATIC)

    def _query(self, template: str, kind: str, required, stale=(), keywords=()) -> None:
        self.queries.append(
            BenchQuery(
                id=f"{self.user_id}-q{len(self.queries):02d}",
                text=template,
                kind=kind,
                required_ids=frozenset(required),
                stale_ids=frozenset(stale),
                answer_keywords=tuple(keywords),
                timestamp=HORIZON_DAYS + 1,
            )
        )

    def _assistant(self, topic: str, day: float, lead: str = "") -> None:
        sentences = TOPICS[topic].sentences
        k = self.rng.randint(min(3, len(sentences)), len(sentences))
        body = " ".join(self.rng.sample(sentences, k))
        opener = lead or self.rng.choice(ACKS)
        self._memory(f"{opener} {body}".strip(), day)

    def _user_fact(self, fact: str, day: float) -> str:
        parts = [fact]
        if self.rng.random() < 0.4:
            parts.insert(0, self.rng.choice(GREETINGS))
        if self.rng.random() < 0.3:
            parts.append(self.rng.choice(SIGNOFFS))
        return self._memory(" ".join(parts), day)

    def _followups(self, topic: str, day: float, max_pairs: int = 2) -> None:
        for _ in range(self.rng.randint(0, max_pairs)):
            self._memory(self.rng.choice(TOPICS[topic].questions), day)
            self._assistant(topic, day)

    def build(self) -> BenchUser:
        rng = self.rng
        events: list[Event] = []

        name = rng.choice(NAMES)
        role = rng.choice(ROLES)
        level = rng.choice(LEVELS)
        languages = rng.choice(LANGUAGE_SETS)
        style = rng.choice(STYLES)

        self._static(f"Name: {name}.")
        self._static(f"Role: {role}.")
        level_id = self._static(f"Experience level: {level}.")
        languages_id = self._static(f"Primary programming languages: {', '.join(languages)}.")
        style_id = self._static(f"Response style preference: {style}.")
        self._static(f"Based in {rng.choice(CITIES)}.")
        self._static(f"Hobbies: {rng.choice(HOBBIES)}. Pets: {rng.choice(PETS)}.")

        self._query(
            rng.choice(BACKGROUND_QUERIES),
            "background",
            required=(languages_id, level_id),
            keywords=(languages[0],),
        )
        self._query(rng.choice(STYLE_QUERIES), "style", required=(style_id,))

        events.extend(self._database_events())
        projects = rng.sample(PROJECTS, rng.randint(2, 3))
        for project in projects:
            events.extend(self._project_events(project, languages))
        for _ in range(rng.randint(45, 65)):
            day = rng.uniform(1, HORIZON_DAYS)
            topic = rng.choice(DISTRACTOR_TOPICS)
            events.append((day, 2, lambda d, t=topic: self._distractor_session(t, d)))

        for day, _, emit in sorted(events, key=lambda e: (e[0], e[1])):
            emit(day)
        self._finalize_queries()
        return BenchUser(self.user_id, self.memories, self.queries)

    def _distractor_session(self, topic: str, day: float) -> None:
        rng = self.rng
        question = rng.choice(TOPICS[topic].questions)
        roll = rng.random()
        if roll < 0.4:
            question = f"{rng.choice(ASIDES)} Anyway, {question[0].lower()}{question[1:]}"
        elif roll < 0.5:
            aside = rng.choice(TECH_ASIDES).format(db=rng.choice(sorted(DATABASES)))
            question = f"{aside} {question}"
        self._memory(question, day)
        self._assistant(topic, day)
        self._followups(topic, day, max_pairs=2)

    def _database_events(self) -> list[Event]:
        rng = self.rng
        initial_db, switched_db = rng.sample(sorted(DATABASES), 2)
        state: dict[str, str] = {}

        def initial(day: float) -> None:
            state["initial"] = self._user_fact(rng.choice(DB_INITIAL).format(db=initial_db), day)
            self._assistant("databases", day)

        def switch(day: float) -> None:
            text = rng.choice(DB_SWITCH).format(
                db=switched_db, old=initial_db, reason=DATABASES[switched_db]
            )
            state["switch"] = self._user_fact(text, day)
            self._assistant("databases", day)
            self._followups("databases", day, max_pairs=1)

        events = [(rng.uniform(3, 60), 0, initial)]
        if rng.random() < 0.6:
            events.append((rng.uniform(90, HORIZON_DAYS - 5), 0, switch))

        def add_query() -> None:
            if "switch" in state:
                required, stale, db = [state["switch"]], [state["initial"]], switched_db
            else:
                required, stale, db = [state["initial"]], [], initial_db
            self._query(rng.choice(DB_QUERIES), "preference", required, stale=stale, keywords=(db,))

        self._pending.append(add_query)
        return events

    def _finalize_queries(self) -> None:
        for add_query in self._pending:
            add_query()

    def _project_events(self, project: str, languages: tuple[str, ...]) -> list[Event]:
        rng = self.rng
        start = rng.uniform(5, 100)
        domain = rng.choice(DOMAINS)
        stack = f"{languages[0]} and {rng.choice(sorted(DATABASES))}"
        state: dict = {"statuses": [], "decisions": {}}
        events = []

        def intro(day: float) -> None:
            text = rng.choice(INTRO_TEMPLATES).format(project=project, domain=domain, stack=stack)
            state["intro"] = self._user_fact(text, day)
            self._assistant("api", day, lead=f"{project} sounds like a fun one.")
            self._followups("databases", day, max_pairs=1)

        events.append((start, 0, intro))

        aspects = rng.sample(sorted(ASPECTS), rng.randint(3, 4))
        for aspect in aspects:
            topic, choices = ASPECTS[aspect]
            choice, alternative, revised = rng.sample(choices, 3)
            decided_day = start + rng.uniform(1, 40)
            record: dict = {"choice": choice}
            state["decisions"][aspect] = record

            if rng.random() < 0.5:

                def deliberate(day, a=aspect, t=topic, c=choice, alt=alternative):
                    text = rng.choice(DELIBERATION_TEMPLATES).format(
                        a=c, b=alt, aspect=a, project=project
                    )
                    self._memory(text, day)
                    self._assistant(t, day)

                events.append((max(start + 0.5, decided_day - rng.uniform(0.5, 5)), 1, deliberate))

            def decide(day, a=aspect, t=topic, c=choice, rec=record):
                text = rng.choice(DECISION_TEMPLATES).format(
                    project=project, choice=c, aspect=a, reason=rng.choice(REASONS)
                )
                rec["id"] = self._user_fact(text, day)
                self._assistant(t, day, lead=f"That sounds like a reasonable call for {project}.")
                self._followups(t, day, max_pairs=1)

            events.append((decided_day, 1, decide))

            revise_day = decided_day + rng.uniform(15, 60)
            if rng.random() < 0.3 and revise_day < HORIZON_DAYS:

                def revise(day, a=aspect, t=topic, old=choice, new=revised, rec=record):
                    text = rng.choice(REVISION_TEMPLATES).format(
                        project=project, aspect=a, old=old, choice=new, reason=rng.choice(REASONS)
                    )
                    rec["stale_id"] = rec["id"]
                    rec["id"] = self._user_fact(text, day)
                    rec["choice"] = new
                    self._assistant(t, day)

                events.append((revise_day, 1, revise))

        statuses = rng.sample(STATUSES, rng.randint(2, 3))
        status_days = sorted(rng.uniform(start + 5, HORIZON_DAYS) for _ in statuses)
        for day, (status, keyword) in zip(status_days, statuses, strict=True):

            def update(d, s=status, kw=keyword):
                text = rng.choice(STATUS_TEMPLATES).format(project=project, status=s)
                state["statuses"].append((self._user_fact(text, d), kw))
                self._assistant(rng.choice(("testing", "deployment", "databases")), d)

            events.append((day, 1, update))

        def add_queries() -> None:
            (latest_status, keyword), *_ = reversed(state["statuses"])
            self._query(
                rng.choice(CONTINUATION_QUERIES).format(project=project),
                "continuation",
                required=(state["intro"], latest_status),
                stale=[sid for sid, _ in state["statuses"][:-1]],
                keywords=(keyword,),
            )
            for aspect in rng.sample(sorted(state["decisions"]), 2):
                record = state["decisions"][aspect]
                self._query(
                    rng.choice(DECISION_QUERIES).format(aspect=aspect, project=project),
                    "decision",
                    required=(record["id"],),
                    stale=[record["stale_id"]] if "stale_id" in record else [],
                    keywords=(record["choice"],),
                )

        self._pending.append(add_queries)
        return events


def generate_user(user_id: str, seed: int = 0) -> BenchUser:
    return _UserBuilder(user_id, random.Random(f"contextio-{seed}-{user_id}")).build()


def generate_dataset(n_users: int = 100, seed: int = 0) -> Dataset:
    """Generate ``n_users`` synthetic users. Output is fully determined by ``seed``."""
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    users = [generate_user(f"u{i:03d}", seed) for i in range(n_users)]
    return Dataset(users, name=f"synthetic-{n_users}u-seed{seed}")
