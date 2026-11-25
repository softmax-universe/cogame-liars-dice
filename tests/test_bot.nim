## The scripted baselines must play whole episodes without ever proposing an
## illegal action — they are both the no-credentials fallback (offline
## certification) and fieldable policies, so this is the completion path. The
## calibrated one must also actually beat the loose one, or the "calibrated"
## label means nothing. And the LLM reply parser must be tolerant of the
## shapes a model really emits without ever letting an illegal move through.

import std/[json, monotimes, os, strutils, times, unicode, unittest]
import liars_dice/[llm, sim]

## Every test in this file is about the SCRIPTED path. Clear the credential
## environment up front so no test can accidentally reach the network.
delEnv("ANTHROPIC_API_KEY")
delEnv("ANTHROPIC_API_KEY_URI")
delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
delEnv("AWS_BEARER_TOKEN_BEDROCK")

proc fixture(seats = 4, deals = 3, seed = 0, mode = mDice, handSize = 5,
    talk = true): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.mode = mode
  result.handSize = handSize
  result.deals = deals
  result.talk = talk
  result.sampled = true
  for index in 0 ..< seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

type Trace = object
  actions: int
  bids: int
  challenges: int

proc playBaselines(config: GameConfig, baselines: seq[string]): (Sim, Trace) =
  ## Drives a whole episode from the scripted baselines alone, asserting that
  ## every action they emit is accepted by the sim FIRST TIME: applyBid or
  ## applyChallenge raising anywhere fails the test.
  let client = newLlmClient(config)
  var sim = initSim(config)
  var trace = Trace()
  while not sim.done:
    let turn = sim.currentTurn()
    case turn.kind
    of tkDeal:
      sim.beginDeal()
    of tkAct:
      let standing = sim.bidSeat >= 0
      let before = sim.bidQuantity
      let decision = client.scriptedAction(sim, turn.seat,
        baselines[turn.seat])
      ## Bounded orders: exactly one action, never talk, never notes.
      inc trace.actions
      check decision.scripted
      check decision.say.len == 0
      check decision.notes.len == 0
      if decision.action == aBid:
        inc trace.bids
        check sim.legalBid(decision.quantity, decision.face)
        check decision.face >= sim.config.lowFace()
        check decision.face <= sim.config.highFace()
        check decision.quantity >= 1
        check decision.quantity <= sim.totalSymbols()
        if standing:
          ## The raise window is q0 .. q0 + 2, plus at most pressure's pad.
          check decision.quantity >= before
          check decision.quantity <= before + 3
        sim.applyBid(turn.seat, decision.quantity, decision.face,
          decision.say, decision.notes, decision.scripted)
      else:
        inc trace.challenges
        check standing
        sim.applyChallenge(turn.seat, decision.say, decision.notes,
          decision.scripted)
      check sim.bidsThisDeal <= sim.config.maxBidsPerDeal
    of tkNone:
      discard
  result = (sim, trace)

suite "scripted baselines":
  test "both baselines play legal episodes across seeds, modes, talk, seats":
    let started = getMonoTime()
    for baseline in ["bayes", "pressure"]:
      for seed in [1, 7, 42, 1234]:
        for mode in [mDice, mPoker]:
          for talk in [true, false]:
            for seats in [3, 4, 6]:
              let config = fixture(seats = seats, deals = 3, seed = seed,
                mode = mode, handSize = (if mode == mPoker: 8 else: 5),
                talk = talk)
              var names: seq[string]
              for index in 0 ..< seats:
                names.add(baseline)
              let (sim, trace) = playBaselines(config, names)
              check sim.done
              check sim.reason == "complete"
              check sim.dealsPlayed == 3
              ## One challenge closes each deal, and nothing else can.
              check trace.challenges == 3
              check trace.actions == trace.bids + trace.challenges
              let results = sim.resultsJson()
              for slot in 0 ..< seats:
                check results["scores"][slot].getFloat() >= 0.0
                check results["scores"][slot].getFloat() <= 1.0
              var sum = 0
              for slot in 0 ..< seats:
                sum += results["points"][slot].getInt()
              check sum == 0
    let elapsed = (getMonoTime() - started).inMilliseconds
    echo "baseline sweep: ", elapsed, " ms"
    check elapsed < 240_000

  test "only explicit bayes and pressure scripted baselines are accepted":
    check parseBaseline("pressure") == "pressure"
    check parseBaseline("bayes") == "bayes"
    for bad in ["PRESSURE", "1", "", "nonsense"]:
      expect LiarsDiceError:
        discard parseBaseline(bad)
    let config = fixture(seed = 5)
    let client = newLlmClient(config)
    var sim = initSim(config)
    sim.beginDeal()
    let seat = sim.currentTurn().seat
    expect LiarsDiceError:
      discard client.scriptedAction(sim, seat, "nonsense")

  test "calibration: two bayes seats beat two pressure seats":
    var bayesTotal = 0.0
    var pressureTotal = 0.0
    var samples = 0
    for seed in [1, 7, 42, 1234]:
      let config = fixture(seats = 4, deals = 30, seed = seed)
      let (sim, _) = playBaselines(config,
        @["bayes", "pressure", "bayes", "pressure"])
      check sim.dealsPlayed == 30
      for slot in 0 ..< 4:
        let score = sim.score(slot)
        check score >= 0.0
        check score <= 1.0
        if slot mod 2 == 0: bayesTotal += score else: pressureTotal += score
      samples += 2
    let bayesMean = bayesTotal / samples.float
    let pressureMean = pressureTotal / samples.float
    echo "bayes mean ", bayesMean, " vs pressure mean ", pressureMean
    check bayesMean > 0.5
    check pressureMean < 0.5

  test "the raise enumeration never exceeds 3 x faces candidates":
    ## The bound `RaiseQuantitySteps * faces` holds by construction, which is
    ## exactly why nothing was checking it. Count what the baseline actually
    ## iterates, on real mid-deal states, in both modes.
    var counted = 0
    for seed in [1, 7, 42]:
      for mode in [mDice, mPoker]:
        let config = fixture(seats = 4, deals = 4, seed = seed, mode = mode,
          handSize = (if mode == mPoker: 8 else: 5))
        let client = newLlmClient(config)
        let ceiling = RaiseQuantitySteps * config.faces()
        var widest = 0
        var sim = initSim(config)
        while not sim.done:
          let turn = sim.currentTurn()
          case turn.kind
          of tkDeal:
            sim.beginDeal()
          of tkAct:
            if sim.bidSeat >= 0:
              var candidates = 0
              for candidate in sim.raiseCandidates():
                inc candidates
                ## Each one is inside the window the design names, so the
                ## count cannot be held down by a candidate out of bounds.
                check candidate.quantity >= sim.bidQuantity
                check candidate.quantity <= sim.bidQuantity +
                  RaiseQuantitySteps - 1
                check candidate.face >= sim.config.lowFace()
                check candidate.face <= sim.config.highFace()
              check candidates <= ceiling
              widest = max(widest, candidates)
              inc counted
            let decision = client.scriptedAction(sim, turn.seat, "bayes")
            if decision.action == aBid:
              sim.applyBid(turn.seat, decision.quantity, decision.face)
            else:
              sim.applyChallenge(turn.seat)
          of tkNone:
            discard
        ## The enumeration really does reach its ceiling in this mode —
        ## otherwise the bound above would be vacuous.
        check widest == ceiling
    ## And the mid-deal states really were visited.
    check counted > 0

suite "the LLM path":
  test "decide fails without fabricating a move when no credentials exist":
    let config = fixture(seed = 3)
    let client = newLlmClient(config)
    check client.disabled
    var sim = initSim(config)
    sim.beginDeal()
    let seat = sim.currentTurn().seat
    let started = getMonoTime()
    let decision = client.decide(sim, seat, "bluff every deal",
      scripted = false)
    let elapsed = (getMonoTime() - started).inMilliseconds
    check elapsed < 1000
    check decision.failed
    check not decision.scripted
    check decision.failureReason.len > 0

  test "userPrompt only shows legal action forms":
    let config = fixture(seed = 4)
    var sim = initSim(config)
    sim.beginDeal()
    let opener = sim.currentTurn().seat
    let openPrompt = sim.userPrompt(opener, "")
    check "\"action\":\"bid\"" in openPrompt
    check "\"action\":\"challenge\"" notin openPrompt
    sim.applyBid(opener, sim.totalSymbols(), sim.config.highFace())
    let next = sim.currentTurn().seat
    let maxBidPrompt = sim.userPrompt(next, "")
    check "\"action\":\"challenge\"" in maxBidPrompt
    check "\"action\":\"bid\"" notin maxBidPrompt
    check "\"quantity\":" & $(sim.totalSymbols() + 1) notin maxBidPrompt

  test "replies parse strictly and reject aliases, fences, and non-integers":
    let config = fixture(seed = 8)
    var sim = initSim(config)
    sim.beginDeal()
    let seat = sim.currentTurn().seat
    let validBid = parseReply(sim, extractJsonObject("""{"action":"bid","quantity":3,"face":2}"""))
    check validBid.action == aBid
    check validBid.quantity == 3
    check validBid.face == 2
    let validChallenge = parseReply(sim, extractJsonObject("""{"action":"challenge"}"""))
    check validChallenge.action == aChallenge
    ## Fenced or prose-wrapped JSON is rejected.
    expect LiarsDiceError:
      discard extractJsonObject("```json\n{\"action\":\"challenge\"}\n```")
    expect LiarsDiceError:
      discard extractJsonObject("Here is my move: {\"action\":\"challenge\"}")
    ## Obsolete aliases and case-folded actions are rejected.
    for alias in ["BID", "Bid", "raise", "Challenge", "CALL", "call", "liar", "doubt", "ponder"]:
      expect LiarsDiceError:
        discard parseReply(sim, %*{"action": alias, "quantity": 3, "face": 2})
    ## Numeric strings and floats are rejected; quantity and face must be JSON integers.
    expect LiarsDiceError:
      discard parseReply(sim, %*{"action": "bid", "quantity": "4", "face": 6})
    expect LiarsDiceError:
      discard parseReply(sim, %*{"action": "bid", "quantity": 4, "face": "6"})
    expect LiarsDiceError:
      discard parseReply(sim, %*{"action": "bid", "quantity": 4.5, "face": 6})
    ## Missing action or missing bid fields are rejected.
    expect LiarsDiceError:
      discard parseReply(sim, %*{"quantity": 3, "face": 2})
    expect LiarsDiceError:
      discard parseReply(sim, %*{"action": "bid", "face": 2})
    ## Oversized free text is truncated on rune boundaries.
    var longSay = ""
    for index in 0 ..< 400:
      longSay.add("é")
    var longNotes = ""
    for index in 0 ..< 900:
      longNotes.add("字")
    let padded = parseReply(sim, %*{"action": "challenge", "say": longSay,
      "notes": longNotes})
    check padded.say.runeLen == MaxSayLen
    check padded.notes.runeLen == MaxNotesLen
    check validateUtf8(padded.say) == -1
    check validateUtf8(padded.notes) == -1
    ## An opening bid stands, and then only a strict raise survives the probe.
    sim.applyBid(seat, 5, 3)
    let next = sim.currentTurn().seat
    for bad in [(5, 3), (5, 2), (4, 6), (0, 3), (21, 3), (6, 9)]:
      let reply = parseReply(sim, %*{"action": "bid", "quantity": bad[0],
        "face": bad[1]})
      var probe = sim
      expect LiarsDiceError:
        probe.applyBid(next, reply.quantity, reply.face)
    let good = parseReply(sim, %*{"action": "bid", "quantity": 5, "face": 4})
    var probe = sim
    probe.applyBid(next, good.quantity, good.face)
    check probe.bidQuantity == 5
    check probe.bidFace == 4
    ## A challenge with no standing bid is rejected the same way.
    var fresh = initSim(config)
    fresh.beginDeal()
    let opener = fresh.currentTurn().seat
    let call = parseReply(fresh, %*{"action": "challenge"})
    check call.action == aChallenge
    var freshProbe = fresh
    expect LiarsDiceError:
      freshProbe.applyChallenge(opener)

  test "talk off strips a say out of the parsed reply":
    let config = fixture(seed = 9, talk = false)
    var sim = initSim(config)
    sim.beginDeal()
    let reply = parseReply(sim, %*{"action": "bid", "quantity": 2, "face": 4,
      "say": "twos are cheap", "notes": "keep"})
    check reply.say == ""
    check reply.notes == "keep"
