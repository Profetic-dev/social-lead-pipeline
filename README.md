# Social Lead Pipeline

An AI-augmented system for extracting qualified leads from social media noise.

Built by [PROFETIC](https://profetic.dev)

---

## Overview

This pipeline transforms raw social media data into qualified, actionable leads through a multi-stage classification system combining rule-based filtering with AI-powered contextual analysis.

```
Social Data → Scraper → Deduplication → Rules Engine → AI Analysis → Scoring → Outreach Queue
     ↓            ↓           ↓              ↓              ↓           ↓            ↓
  API fetch    Paginate    Hash + fuzzy   16 DQ rules    Context    3 tiers    DM-ready
  + queries    + rate      matching       fast filter    + extract   ranked     filtered
               limit                                     entities
```

**Performance:** 30% qualification rate from raw social data, with tiered confidence scoring.

---

## Architecture

### 1. Scraper Module

Configurable ingestion with query optimization and persistence:

```python
class TweetScraper:
    """
    Multi-query scraper with rate limiting and incremental pulls.
    """
    
    def __init__(self, api_key: str, history_file: str = 'ngram_history.json'):
        self.api_key = api_key
        self.history_file = history_file
        self.seen_ids = self._load_history()
    
    def search(self, query: str, pages: int = 5) -> List[Dict]:
        """
        Execute search with automatic pagination and deduplication.
        
        Queries support boolean operators:
        - "got into a car accident" lang:en -filter:retweets
        - "rear ended me" OR "t-boned" lang:en -almost -nearly
        """
        results = []
        cursor = None
        
        for page in range(pages):
            response = self._api_call(query, cursor)
            
            for tweet in response.get('tweets', []):
                # Skip if already seen (incremental scraping)
                if tweet['id'] in self.seen_ids:
                    continue
                    
                # Extract relevant fields
                results.append({
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'author_id': tweet['author']['id'],
                    'author_username': tweet['author']['userName'],
                    'created_at': tweet['createdAt'],
                    'is_reply': tweet.get('isReply', False),
                    'conversation_id': tweet.get('conversationId'),
                    # DM availability for outreach
                    'can_dm': not tweet['author'].get('privateDM', True),
                })
                
                self.seen_ids.add(tweet['id'])
            
            cursor = response.get('next_cursor')
            if not cursor:
                break
                
            # Rate limiting
            time.sleep(1.5)
        
        self._save_history()
        return results
    
    def _load_history(self) -> Set[str]:
        """Load seen IDs for incremental scraping across runs."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return set(data.get('seen_ids', []))
        return set()
```

### 2. Deduplication Engine

Multi-layer duplicate detection:

```python
class Deduplicator:
    """
    Hash-based + fuzzy matching deduplication.
    Catches exact duplicates, retweets, and near-duplicates.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.seen_hashes = set()
        self.seen_texts = []
    
    def is_duplicate(self, tweet: Dict) -> bool:
        text = tweet['text']
        
        # Layer 1: Exact hash match
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        # Layer 2: Normalized hash (removes URLs, mentions, extra whitespace)
        normalized = self._normalize(text)
        norm_hash = hashlib.md5(normalized.encode()).hexdigest()
        if norm_hash in self.seen_hashes:
            return True
        
        # Layer 3: Fuzzy match against recent texts
        for seen_text in self.seen_texts[-1000:]:  # Rolling window
            similarity = self._similarity(normalized, seen_text)
            if similarity > self.threshold:
                return True
        
        # Not a duplicate — record it
        self.seen_hashes.add(text_hash)
        self.seen_hashes.add(norm_hash)
        self.seen_texts.append(normalized)
        return False
    
    def _normalize(self, text: str) -> str:
        """Remove noise for comparison."""
        text = re.sub(r'https?://\S+', '', text)      # URLs
        text = re.sub(r'@\w+', '', text)              # Mentions
        text = re.sub(r'\s+', ' ', text).strip()      # Whitespace
        return text.lower()
    
    def _similarity(self, a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union
```

### 3. Rules Engine (16 Disqualification Rules)

Fast, deterministic filtering before expensive AI calls:

```python
class RulesEngine:
    """
    16-rule disqualification engine.
    Each rule has patterns + optional context requirements.
    """
    
    # Rule definitions (subset shown)
    RULES = {
        'at_fault': {
            'patterns': [
                r'\bi\s+(hit|ran into|rear.?ended|crashed into)\b',
                r'\bmy fault\b',
                r'\bi was (speeding|texting|drinking)\b',
            ],
            'description': 'User caused the incident'
        },
        'total_loss': {
            'patterns': [
                r'\btotaled\b',
                r'\btotal loss\b', 
                r'\bcar is gone\b',
                r'\bwrite.?off\b',
            ],
            'description': 'Vehicle destroyed, no residual value'
        },
        'hit_and_run': {
            'patterns': [
                r'\bhit and run\b',
                r'\bdrove off\b',
                r'\bfled the scene\b',
                r'\bnever found\b.{0,30}\b(driver|person|car)\b',
            ],
            'description': 'No identifiable at-fault party'
        },
        'fatality': {
            'patterns': [
                r'\b(killed|died|passed away|fatal)\b',
                r'\brest in (peace|heaven)\b',
                r'\bRIP\b',
            ],
            'description': 'Fatality involved'
        },
        'news_account': {
            'patterns': [
                r'\b(news|reporter|journalist|anchor)\b',
                r'\bbreaking\s*:\b',
            ],
            'check_bio': True,
            'bio_patterns': [r'\bnews\b', r'\breporter\b', r'\bjournalist\b'],
            'description': 'News outlet, not individual'
        },
        'no_fault_state': {
            'patterns': [],
            'location_check': ['MI', 'michigan', 'NE', 'nebraska'],
            'description': 'No-fault insurance state'
        },
        'hypothetical': {
            'patterns': [
                r'\bwhat if\b',
                r'\bwould you\b',
                r'\bimagine\b',
                r'\bhypothetically\b',
            ],
            'description': 'Not describing real event'
        },
        'near_miss': {
            'patterns': [
                r'\balmost\b',
                r'\bnearly\b',
                r'\bclose call\b',
                r'\bjust missed\b',
            ],
            'description': 'No actual collision'
        },
        'third_party': {
            'patterns': [
                r'\bmy (friend|brother|sister|mom|dad|cousin)\b',
                r'\bsomeone i know\b',
            ],
            'check_context': True,
            'description': 'Describing someone else\'s accident'
        },
        'dm_unavailable': {
            'patterns': [],
            'check_field': 'can_dm',
            'expected_value': False,
            'description': 'Cannot contact via DM'
        },
        # ... additional rules ...
    }
    
    def evaluate(self, tweet: Dict) -> Tuple[bool, List[str]]:
        """
        Run all rules against tweet.
        Returns: (is_qualified, list_of_triggered_rules)
        """
        triggered = []
        text = tweet['text'].lower()
        
        for rule_name, rule in self.RULES.items():
            # Pattern matching
            for pattern in rule.get('patterns', []):
                if re.search(pattern, text, re.IGNORECASE):
                    triggered.append(rule_name)
                    break
            
            # Bio check (for news accounts, lawyers, etc.)
            if rule.get('check_bio') and 'author_bio' in tweet:
                bio = tweet.get('author_bio', '').lower()
                for bio_pattern in rule.get('bio_patterns', []):
                    if re.search(bio_pattern, bio):
                        triggered.append(rule_name)
                        break
            
            # Field check (for DM availability, etc.)
            if 'check_field' in rule:
                field_value = tweet.get(rule['check_field'])
                if field_value == rule['expected_value']:
                    triggered.append(rule_name)
            
            # Location check (for no-fault states)
            if 'location_check' in rule:
                location = tweet.get('author_location', '').lower()
                for loc in rule['location_check']:
                    if loc.lower() in location:
                        triggered.append(rule_name)
                        break
        
        is_qualified = len(triggered) == 0
        return is_qualified, triggered
```

### 4. AI Analysis Layer

LLM-powered analysis for nuanced cases that pass rule filtering:

```python
class AIClassifier:
    """
    LLM-based classifier for context-dependent qualification.
    Only called for tweets that pass rules engine.
    """
    
    SYSTEM_PROMPT = """You are a lead qualification specialist. 
    Analyze social media posts to determine if the author experienced a 
    vehicle accident where they were NOT at fault.
    
    QUALIFY if:
    - First-person account of being hit by another driver
    - Clear not-at-fault scenario (rear-ended, t-boned, red light runner hit them)
    - Event appears recent (within last 2 years)
    - Author is the vehicle owner/driver, not a bystander
    
    DISQUALIFY if:
    - Author caused the accident
    - Hypothetical, joke, or fictional scenario
    - News report, not personal account
    - Vehicle was totaled (no residual value to claim)
    - Describing someone else's accident (not their own)
    - Sarcasm or ironic tone
    
    Respond with JSON only."""
    
    OUTPUT_SCHEMA = {
        "qualified": "boolean",
        "confidence": "float 0-1",
        "reasoning": "string explaining decision",
        "extracted": {
            "incident_date": "ISO date or null",
            "location": "string or null",
            "fault_party": "string describing who was at fault",
            "vehicle_status": "driveable | totaled | unknown",
            "incident_type": "rear-end | t-bone | red-light | other"
        }
    }
    
    def classify(self, tweet: Dict) -> Dict:
        """
        Classify a single tweet using LLM.
        Returns structured assessment with confidence score.
        """
        prompt = f"""Analyze this social media post:

Author: @{tweet['author_username']}
Posted: {tweet['created_at']}
Text: {tweet['text']}

Determine if this person experienced a not-at-fault vehicle accident.
Respond with JSON matching this schema: {json.dumps(self.OUTPUT_SCHEMA)}"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            # Validate required fields
            assert 'qualified' in result
            assert 'confidence' in result
            return result
        except (json.JSONDecodeError, AssertionError):
            # Fallback for malformed responses
            return {
                'qualified': False,
                'confidence': 0.0,
                'reasoning': 'Failed to parse AI response',
                'extracted': {}
            }
    
    def batch_classify(self, tweets: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        Classify multiple tweets with batching for efficiency.
        """
        results = []
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            for tweet in batch:
                result = self.classify(tweet)
                result['tweet_id'] = tweet['id']
                results.append(result)
            # Rate limiting between batches
            time.sleep(2)
        return results
```

### 5. Scoring & Tiering

Multi-factor scoring for lead prioritization:

```python
class LeadScorer:
    """
    Score qualified leads for outreach prioritization.
    """
    
    TIER_THRESHOLDS = {
        'super_qualified': 0.85,
        'standard': 0.60,
        'borderline': 0.40,
    }
    
    def score(self, tweet: Dict, ai_result: Dict) -> Dict:
        """
        Calculate composite score from multiple signals.
        """
        score = 0.0
        factors = []
        
        # AI confidence (40% weight)
        ai_confidence = ai_result.get('confidence', 0.5)
        score += ai_confidence * 0.40
        factors.append(f"AI confidence: {ai_confidence:.2f}")
        
        # Recency (20% weight)
        days_old = self._days_since(tweet['created_at'])
        recency_score = max(0, 1 - (days_old / 365))  # Decay over 1 year
        score += recency_score * 0.20
        factors.append(f"Recency: {recency_score:.2f} ({days_old} days)")
        
        # Engagement signals (15% weight)
        # Higher engagement = more visible = more likely real
        engagement = tweet.get('likes', 0) + tweet.get('retweets', 0)
        engagement_score = min(1.0, engagement / 50)  # Cap at 50
        score += engagement_score * 0.15
        factors.append(f"Engagement: {engagement_score:.2f}")
        
        # DM availability (15% weight)
        can_dm = tweet.get('can_dm', False)
        dm_score = 1.0 if can_dm else 0.0
        score += dm_score * 0.15
        factors.append(f"DM available: {can_dm}")
        
        # Clear not-at-fault indicator (10% weight)
        incident_type = ai_result.get('extracted', {}).get('incident_type', 'other')
        clear_types = ['rear-end', 't-bone', 'red-light']
        type_score = 1.0 if incident_type in clear_types else 0.5
        score += type_score * 0.10
        factors.append(f"Incident type: {incident_type}")
        
        # Determine tier
        if score >= self.TIER_THRESHOLDS['super_qualified']:
            tier = 'super_qualified'
        elif score >= self.TIER_THRESHOLDS['standard']:
            tier = 'standard'
        elif score >= self.TIER_THRESHOLDS['borderline']:
            tier = 'borderline'
        else:
            tier = 'disqualified'
        
        return {
            'score': round(score, 3),
            'tier': tier,
            'factors': factors,
            'dm_available': can_dm,
        }
```

### 6. Outreach Queue Generation

Final output formatted for CRM integration:

```python
class OutreachGenerator:
    """
    Generate prioritized outreach queue from scored leads.
    """
    
    def generate(self, leads: List[Dict], output_path: str) -> None:
        """
        Create outreach queue CSV/Excel with all relevant data.
        """
        rows = []
        
        for lead in sorted(leads, key=lambda x: x['score'], reverse=True):
            # Skip if can't DM
            if not lead.get('dm_available', False):
                continue
            
            rows.append({
                'tier': lead['tier'],
                'score': lead['score'],
                'username': f"@{lead['author_username']}",
                'tweet_url': f"https://twitter.com/{lead['author_username']}/status/{lead['tweet_id']}",
                'tweet_text': lead['text'][:280],
                'incident_type': lead.get('extracted', {}).get('incident_type', ''),
                'incident_date': lead.get('extracted', {}).get('incident_date', ''),
                'location': lead.get('extracted', {}).get('location', ''),
                'ai_reasoning': lead.get('reasoning', ''),
                'scraped_at': lead['created_at'],
            })
        
        # Export
        df = pd.DataFrame(rows)
        
        if output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        # Summary stats
        print(f"Generated outreach queue: {len(rows)} leads")
        print(f"  🔥 Super Qualified: {len([r for r in rows if r['tier'] == 'super_qualified'])}")
        print(f"  ✅ Standard: {len([r for r in rows if r['tier'] == 'standard'])}")
        print(f"  ⚠️ Borderline: {len([r for r in rows if r['tier'] == 'borderline'])}")
```

---

## Pipeline Flow

```python
def run_pipeline(queries: List[str], output_path: str):
    """
    Full pipeline execution.
    """
    # 1. Scrape
    scraper = TweetScraper(api_key=os.environ['TWITTER_API_KEY'])
    raw_tweets = []
    for query in queries:
        raw_tweets.extend(scraper.search(query, pages=5))
    print(f"Scraped: {len(raw_tweets)} tweets")
    
    # 2. Deduplicate
    deduper = Deduplicator()
    unique_tweets = [t for t in raw_tweets if not deduper.is_duplicate(t)]
    print(f"After dedup: {len(unique_tweets)} tweets")
    
    # 3. Rules filter (fast, cheap)
    rules = RulesEngine()
    passed_rules = []
    for tweet in unique_tweets:
        qualified, triggered = rules.evaluate(tweet)
        if qualified:
            passed_rules.append(tweet)
        else:
            tweet['dq_rules'] = triggered
    print(f"Passed rules: {len(passed_rules)} tweets")
    
    # 4. AI classification (slow, expensive — only for survivors)
    ai = AIClassifier()
    ai_results = ai.batch_classify(passed_rules)
    
    # 5. Merge and score
    scorer = LeadScorer()
    leads = []
    for tweet, ai_result in zip(passed_rules, ai_results):
        if ai_result['qualified']:
            score_result = scorer.score(tweet, ai_result)
            lead = {**tweet, **ai_result, **score_result}
            leads.append(lead)
    print(f"Qualified leads: {len(leads)}")
    
    # 6. Generate outreach queue
    generator = OutreachGenerator()
    generator.generate(leads, output_path)
```

---

## Outreach Automation

Automated DM delivery with batch management and response tracking.

### Batch Manager

State-tracked outreach with pause/resume capability:

```python
# Check current status
python scripts/batch_manager.py status

# Create next batch of 50 leads
python scripts/batch_manager.py next 50

# Mark batch complete
python scripts/batch_manager.py complete
```

**State file:** `outreach_state.json` tracks:
- Current batch number and status
- Total leads sent / remaining
- Next row to process from master file
- Batch history with timestamps

### TwBoost Integration

Chrome extension for automated DM delivery:
- CSV import with username + tweet URL
- Configurable delays (1-2 min recommended)
- Batch pausing (every 50 messages)
- Skip already-contacted users
- Response tracking built-in

**Settings for safe operation:**
| Setting | Recommended |
|---------|-------------|
| Min delay | 1 minute |
| Max delay | 2-3 minutes |
| Batch pause after | 50 messages |
| Pause duration | 10-15 minutes |

### Call Tracking (RingCentral API)

For clients using RingCentral, webhook integration provides:
- Real-time call notifications
- Caller number matching against lead list
- No phone number changes required
- Full attribution without trust dependency

---

## Results

Deployed in production for diminished value lead generation.

| Metric | Value |
|--------|-------|
| Raw → Qualified rate | 30% |
| Rules filter efficiency | Removes 70% via 17 DQ rules |
| Processing speed | 500 tweets/minute |
| DM-available rate | 100% (Rule 17 filters closed DMs) |
| Outreach capacity | ~20,000 DMs/month |

### Production Stats (Feb 2026)

| Metric | Value |
|--------|-------|
| Tweets scraped | 7,095 |
| Qualified leads | 2,098 |
| Qualification rate | 29.6% |

### Lead Distribution

| Tier | % of Qualified | Description |
|------|---------------|-------------|
| 🔥 Super Qualified | 18% | High confidence, recent, clear scenario |
| ✅ Standard | 75% | Good confidence, actionable |
| ⚠️ Borderline | 7% | Lower confidence, may need manual review |

---

## Query Optimization (N-gram Tracker)

Automated discovery of new high-value search queries:

```bash
# Process classified leads and update tracker
python skills/dv-classifier/scripts/ngram_tracker.py \
  --input classified/*.csv

# View query candidates
python skills/dv-classifier/scripts/ngram_tracker.py --candidates
```

Scores phrases based on:
- Frequency across all historical data
- DV relevance (accident, crash, fault, insurance keywords)
- Phrase length (prefers 3-5 word phrases)
- Fault/blame language bonus

---

## Tech Stack

- **Language:** Python 3.11+
- **Scraping:** TwitterAPI.io ($0.15/1K tweets)
- **Outreach:** TwBoost ($14/mo unlimited)
- **Call Tracking:** RingCentral API (optional)
- **Data:** pandas, JSON processing
- **Output:** CSV, Excel, JSON

### Monthly Operating Costs

| Item | Cost |
|------|------|
| TwitterAPI.io | ~$10-15 |
| TwBoost Pro | $14 |
| **Total** | **~$25-30** |

---

## Private Repository

This is a showcase repository demonstrating architecture and approach.

Full implementation (including all 16 DQ rules and production configs) available for client engagements.

**Contact:** [hello@profetic.dev](mailto:hello@profetic.dev)

---

© 2026 PROFETIC
