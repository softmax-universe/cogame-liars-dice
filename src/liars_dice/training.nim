## Private authoritative decisions, separate from spectator artifacts.
import std/[json, options]
import bitworld/decision_trajectory
import llm, sim, prompts

proc recordAppliedDecision*(trajectory: DecisionTrajectory, before, after: Sim,
    seat: int, proposed: Decision, operatorPrompt: string,
    deliberateTeacher: bool, fallbackReason = "") =
  var applied = newJNull()
  for index in before.events.len ..< after.events.len:
    let event = after.events[index]
    if event.seat == seat and event.kind in {evBid, evChallenge}:
      let actual = Decision(action: (if event.kind == evBid: aBid else: aChallenge),
        quantity: event.quantity, face: event.face, say: event.say, notes: event.notes)
      applied = before.decisionAction(actual)
      break
  doAssert applied.kind == JObject, "engine emitted no applied decision event"
  var attempts = proposed.nativeAttempts
  var selected = none(string)
  let forcedRule = fallbackReason.len > 0
  if not forcedRule and attempts.len == 0:
    let origin = if deliberateTeacher: aoTeacher else: aoUnknown
    var evidence = newDecisionAttempt("event-" & $before.events.len & "-applied",
      if deliberateTeacher: proposed.policy else: "external-liars-dice", origin)
    if deliberateTeacher:
      evidence.prompt = %*[{"role": "system", "content": systemPrompt(before, seat)},
        {"role": "user", "content": userPrompt(before, seat, operatorPrompt)}]
      evidence.response = %($before.decisionAction(proposed))
    else:
      evidence.response = %proposed.submittedResponse
    evidence.parsedAction = before.decisionAction(proposed)
    evidence.accepted = true
    attempts.add(evidence)
  for index in 0 ..< attempts.len:
    if attempts[index].accepted:
      if forcedRule:
        attempts[index].accepted = false
        attempts[index].rejectionReason = some(fallbackReason)
      else:
        selected = some(attempts[index].attemptId)
  let system = systemPrompt(before, seat)
  let user = userPrompt(before, seat, operatorPrompt)
  let observation = %*{"seat": seat, "system": system, "user": user,
    "prompt_template": promptMetadata(before.config, system, user)}
  trajectory.recordDecision("event-" & $before.events.len, $seat,
    observation, attempts, selected, applied,
    if forcedRule: asFallback else: asAccepted, terminal = after.done,
    fallbackOrigin = if forcedRule: some(fallbackReason) else: none(string))

proc recordFailedDecision*(trajectory: DecisionTrajectory, before: Sim,
    seat: int, proposed: Decision, operatorPrompt: string,
    missing = false) =
  ## Records a genuine policy failure without fabricating any executed action.
  let system = systemPrompt(before, seat)
  let user = userPrompt(before, seat, operatorPrompt)
  let observation = %*{"seat": seat, "system": system, "user": user,
    "prompt_template": promptMetadata(before.config, system, user)}
  var attempts = proposed.nativeAttempts
  if attempts.len == 0 and proposed.submittedResponse.len > 0:
    var evidence = newDecisionAttempt("event-" & $before.events.len & "-rejected",
      "external-liars-dice", aoUnknown)
    evidence.response = %proposed.submittedResponse
    evidence.rejectionReason = some(proposed.failureReason)
    attempts.add(evidence)
  trajectory.recordDecision("event-" & $before.events.len, $seat,
    observation, attempts, none(string), newJNull(),
    if missing: asMissing else: asRejected, terminal = true,
    fallbackOrigin = none(string))

proc finishTrajectory*(trajectory: DecisionTrajectory, sim: Sim) =
  let outcome = sim.resultsJson()
  var participants = newJObject()
  for seat in 0 ..< sim.config.players.len:
    participants[$seat] = %*{"score": outcome["scores"][seat]}
  trajectory.finish(if sim.reason == "complete": esCompleted else: esTruncated,
    outcome, participants)

proc finishFailedTrajectory*(trajectory: DecisionTrajectory, sim: Sim,
    seat: int, reason: string) =
  ## Ends the trajectory as failed without publishing participant scores.
  let outcome = %*{
    "reason": "player_failure",
    "failedSeat": seat,
    "alias": sim.names[seat],
    "deals": sim.dealsPlayed,
    "maxDeals": sim.config.deals,
    "mode": $sim.config.mode,
    "talk": sim.config.talk,
    "error": reason
  }
  trajectory.finish(esFailed, outcome, newJObject())
