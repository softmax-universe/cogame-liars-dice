## Liar's Dice player: canonical native model and scripted baseline runner.
##
## Modes:
## 1. Explicit scripted baseline (`PLAYER_SCRIPTED=bayes|pressure`):
##    Registers its seat as one of the built-in scripted baselines; the server
##    plays it deterministically with no LLM. Any other non-empty value is
##    rejected at startup.
## 2. Ordinary model player (`COWORLD_LLM_ENABLED=true`, or `COWORLD_LLM_ENDPOINT`
##    / `OPENAI_BASE_URL` / `OPENROUTER_API_KEY` set in the player container):
##    Registers as an external controller (`control: "external"`) carrying
##    `PLAYER_PROMPT` (or the default strategy), receives each seat observation,
##    makes one direct HTTP chat-completions call to the container's configured
##    LLM endpoint for `COWORLD_LLM_MODEL`, parses the canonical JSON object
##    (accepting a single Markdown presentation fence without inventing
##    actions), and submits the action with preserved raw response evidence.
##    Errors, refusals, missing endpoints, or invalid replies report an explicit
##    `action_error` without retrying, falling back to another model, or
##    fabricating a scripted move.
## 3. Legacy server-side prompt delivery (neither `PLAYER_SCRIPTED` nor any
##    player-side LLM env configured): delivers `PLAYER_PROMPT` with
##    `scripted: false` so the game server drives the seat if configured.

import
  std/[json, math, options, os, strutils],
  curly,
  whisky,
  liars_dice/llm

const
  DefaultPrompt = """
Count the dice you can see before you speak. Your own hand plus the table's
expected share (one sixth of the dice you cannot see, per face) is your honest
estimate of any face's count; treat any bid more than two above that estimate as
probably a lie. Challenge when you judge the standing bid is under 40% likely to
be true, and never challenge a bid you would happily have made yourself. When
you raise, prefer the smallest legal raise on a face you actually hold - it
costs the least credibility and leaves you room later. Use table talk to build a
picture you can cash in: claim a face you do NOT hold early in the episode, then
bid honestly on it later. Keep in your notes, per opponent, how often their bids
turned out false and whether their talk matched their hands.
"""
  DefaultLlmModel = "anthropic/claude-haiku-4.5"
  DefaultMaxTokens = 900
  DefaultTimeoutSeconds = 30
  OpenRouterCompletionsUrl = "https://openrouter.ai/api/v1/chat/completions"

type
  ModelTransport = object
    url: string
    authHeader: string
    model: string
    maxTokens: int
    timeoutSeconds: int
    temperature: float

proc resolveCompletionUrl(): tuple[url, authHeader: string] =
  let coworldEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  let openaiBase = getEnv("OPENAI_BASE_URL").strip()
  let openaiKey = getEnv("OPENAI_API_KEY").strip()
  let openrouterKey = getEnv("OPENROUTER_API_KEY").strip()
  if coworldEndpoint.len > 0:
    let base = coworldEndpoint.strip(chars = {'/'}, leading = false)
    let url =
      if base.endsWith("/chat/completions"): base
      elif base.endsWith("/v1"): base & "/chat/completions"
      else: base & "/v1/chat/completions"
    let key =
      if openaiKey.len > 0: openaiKey
      elif openrouterKey.len > 0: openrouterKey
      else: ""
    return (url, if key.len > 0: "Bearer " & key else: "")
  if openaiBase.len > 0:
    let base = openaiBase.strip(chars = {'/'}, leading = false)
    let url =
      if base.endsWith("/chat/completions"): base
      else: base & "/chat/completions"
    return (url, if openaiKey.len > 0: "Bearer " & openaiKey else: "")
  if openrouterKey.len > 0:
    return (OpenRouterCompletionsUrl, "Bearer " & openrouterKey)
  ("", "")

proc loadModelTransport(): ModelTransport =
  let (url, authHeader) = resolveCompletionUrl()
  let model = getEnv("COWORLD_LLM_MODEL").strip()
  let tempText = getEnv("COWORLD_LLM_TEMPERATURE", "1").strip()
  let temp = parseFloat(tempText)
  if temp < 0 or temp > 1 or classify(temp) in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError,
      "COWORLD_LLM_TEMPERATURE must be finite and between zero and one")
  let maxTok =
    try: parseInt(getEnv("COWORLD_LLM_MAX_TOKENS", $DefaultMaxTokens).strip())
    except ValueError: DefaultMaxTokens
  let timeoutSec =
    try: parseInt(getEnv("COWORLD_LLM_TIMEOUT_SECONDS", $DefaultTimeoutSeconds).strip())
    except ValueError: DefaultTimeoutSeconds
  ModelTransport(
    url: url,
    authHeader: authHeader,
    model: if model.len > 0: model else: DefaultLlmModel,
    maxTokens: max(64, maxTok),
    timeoutSeconds: max(5, timeoutSec),
    temperature: temp
  )

proc extractMessageText(choice: JsonNode): string =
  let message = choice{"message"}
  if message.isNil or message.kind != JObject:
    raise newException(ValueError, "chat completion choice missing message object")
  let refusal = message{"refusal"}
  if not refusal.isNil and refusal.kind == JString and refusal.getStr().strip().len > 0:
    raise newException(ValueError, "model refusal: " & refusal.getStr().strip())
  let content = message{"content"}
  if content.isNil or content.kind == JNull:
    raise newException(ValueError, "chat completion message content is empty")
  if content.kind == JString:
    return content.getStr()
  if content.kind == JArray:
    for part in content:
      if part.kind == JObject and part{"type"}.getStr() == "text":
        result.add(part{"text"}.getStr())
    return
  raise newException(ValueError, "unsupported chat completion content type")

proc callModelOnce(curl: Curly, transport: ModelTransport, slot: int,
    systemPrompt, userPrompt: string,
    rawBody: var string): tuple[action: JsonNode, servedModel: string] =
  if transport.url.len == 0:
    raise newException(ValueError, "no LLM endpoint or credentials configured")
  let body = %*{
    "model": transport.model,
    "messages": [
      {"role": "system", "content": systemPrompt},
      {"role": "user", "content": userPrompt}
    ],
    "max_tokens": transport.maxTokens,
    "temperature": transport.temperature
  }
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  headers["accept"] = "application/json"
  if slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  if transport.authHeader.len > 0:
    headers["authorization"] = transport.authHeader
  let response = curl.post(transport.url, headers, $body, transport.timeoutSeconds)
  rawBody = response.body
  if response.code < 200 or response.code >= 300:
    let snippet =
      if response.body.len > 300: response.body[0 ..< 300] & "..."
      else: response.body
    raise newException(ValueError,
      "llm HTTP " & $response.code & ": " & snippet.replace("\n", " "))
  let parsed = parseJson(response.body)
  if parsed.kind != JObject:
    raise newException(ValueError, "llm response is not a JSON object")
  result.servedModel = parsed{"model"}.getStr(transport.model)
  let choices = parsed{"choices"}
  if choices.isNil or choices.kind != JArray or choices.len == 0:
    raise newException(ValueError, "llm response missing choices")
  let choice = choices[0]
  if choice{"finish_reason"}.getStr() == "content_filter":
    raise newException(ValueError, "model refusal (content_filter)")
  let text = extractMessageText(choice)
  result.action = extractJsonObject(text)

when isMainModule:
  let url = getEnv("COWORLD_PLAYER_WS_URL")
  if url.len == 0:
    quit("COWORLD_PLAYER_WS_URL is not set", 1)
  var prompt = getEnv("PLAYER_PROMPT")
  if prompt.len == 0:
    prompt = DefaultPrompt
  let requested = getEnv("PLAYER_SCRIPTED").strip()
  let scripted = requested.len > 0
  let baseline =
    if not scripted: ""
    elif requested == "bayes" or requested == "pressure": requested
    else: quit("PLAYER_SCRIPTED must be 'bayes' or 'pressure', got: " & requested, 1)

  let llmEnabled = getEnv("COWORLD_LLM_ENABLED").strip().toLowerAscii() == "true"
  let hasEndpointEnv =
    getEnv("COWORLD_LLM_ENDPOINT").strip().len > 0 or
    getEnv("OPENAI_BASE_URL").strip().len > 0 or
    getEnv("OPENROUTER_API_KEY").strip().len > 0
  let modelMode = (not scripted) and (llmEnabled or hasEndpointEnv)
  var transport: ModelTransport
  var curl: Curly
  if modelMode:
    try:
      transport = loadModelTransport()
    except CatchableError as error:
      quit("invalid LLM configuration: " & error.msg, 1)
    curl = newCurly()

  proc initialFrame(): string =
    if scripted:
      $ %*{"type": "prompt", "prompt": prompt,
        "scripted": true, "baseline": baseline}
    elif modelMode:
      $ %*{"type": "register", "control": "external", "prompt": prompt}
    else:
      $ %*{"type": "prompt", "prompt": prompt, "scripted": false}

  echo "liars-dice player: connecting to game"
  let socket = newWebSocket(url)
  var seatSlot = -1

  ## whisky RAISES on a close frame or a truncated read (only a timeout
  ## returns none), and the game's quit(0) can outrun the flushed final
  ## frame — so a dead socket must exit 0, not crash the container.
  try:
    while true:
      let received = socket.receiveMessage()
      if received.isNone:
        echo "liars-dice player: connection closed, exiting"
        break
      let message = received.get()
      if message.kind != TextMessage:
        continue
      try:
        let payload = parseJson(message.data)
        case payload{"type"}.getStr()
        of "welcome":
          seatSlot = payload{"slot"}.getInt(-1)
          echo "liars-dice player: seated at slot ",
            seatSlot, " as ", payload{"name"}.getStr()
          socket.send(initialFrame())
          if scripted:
            echo "liars-dice player: registered scripted baseline ", baseline
          elif modelMode:
            echo "liars-dice player: registered model player (", transport.model,
              ", ", prompt.len, " char prompt)"
          else:
            echo "liars-dice player: prompt delivered (", prompt.len, " chars)"
        of "observation":
          if not modelMode:
            continue
          let eventId = payload["event"].getInt()
          let obs = payload["observation"]
          let inputNode = obs["input"]
          let sysText = inputNode["system"].getStr()
          let userText = inputNode["user"].getStr()
          var rawBody = ""
          try:
            let reply = callModelOnce(curl, transport, seatSlot, sysText, userText, rawBody)
            socket.send($ %*{
              "type": "action",
              "event": eventId,
              "action": reply.action,
              "raw_response": rawBody,
              "model": reply.servedModel
            })
          except CatchableError as error:
            echo "liars-dice player: model call failed on event ", eventId, ": ", error.msg
            socket.send($ %*{
              "type": "action_error",
              "event": eventId,
              "error": error.msg,
              "raw_response": rawBody
            })
        of "final":
          if payload.hasKey("scores"):
            echo "liars-dice player: final scores ", payload{"scores"}
          else:
            echo "liars-dice player: ended (", payload{"reason"}.getStr(), ")"
          break
        else:
          discard
      except CatchableError as error:
        echo "liars-dice player: ignoring bad frame: ", error.msg
  except CatchableError as error:
    echo "liars-dice player: socket closed (", error.msg, "), exiting"
  try:
    socket.close()
  except CatchableError:
    discard
