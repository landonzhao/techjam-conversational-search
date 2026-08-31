"""Must-pass tests for correction-aware preference state and effective queries."""
from __future__ import annotations

from src.dialogue import ConversationState
from src.understanding import Constraint, NeedModel, SlotFiller


def _parse(message: str) -> tuple[NeedModel, ConversationState]:
    need = NeedModel()
    need.revise(SlotFiller().parse(message, turn=1))
    state = ConversationState(user_profile={})
    state.accumulate(message, turn=1)
    state.need = need
    return need, state


def _values(need: NeedModel, slot: str, polarity: int = 1) -> list[str]:
    return [c.value for c in need.constraints if c.slot == slot and c.polarity == polarity]


def test_exclusive_set_overwrites_but_ledger_keeps_audit_history():
    need = NeedModel()
    need.revise([
        Constraint("color", "red", operation="SET"),
        Constraint("color", "blue", operation="SET"),
        Constraint("color", "yellow", operation="SET"),
    ])
    assert _values(need, "color") == ["yellow"]
    assert [e.value for e in need.ledger] == ["red", "blue", "yellow"]
    assert [e.active for e in need.ledger] == [False, False, True]


def test_add_preserves_an_explicit_color_conjunction():
    need, _ = _parse("red and blue")
    assert _values(need, "color") == ["red", "blue"]
    assert [e.operation for e in need.ledger] == ["SET", "ADD"]


def test_remove_clear_and_no_preference_operations():
    need = NeedModel()
    need.revise([Constraint("color", "red", operation="SET")])
    need.revise([Constraint("color", "red", operation="REMOVE")])
    assert not _values(need, "color")
    assert _values(need, "color", polarity=-1) == ["red"]

    need.revise([Constraint("color", "", polarity=0, operation="CLEAR")])
    assert not [c for c in need.constraints if c.slot == "color"]
    assert "color" not in need.no_preference

    need.revise([Constraint("color", "blue", operation="SET")])
    need.revise([Constraint("color", "", polarity=0, operation="NO_PREFERENCE")])
    assert not [c for c in need.constraints if c.slot == "color"]
    assert "color" in need.no_preference


def test_repair_cues_apply_updates_in_character_order():
    need, _ = _parse("red—wait, no, blue—actually yellow is better")
    assert _values(need, "color") == ["yellow"]
    assert [e.value for e in need.ledger] == ["red", "blue", "yellow"]


def test_negative_then_positive_is_not_mistaken_for_a_repair():
    need, _ = _parse("not red, blue")
    assert _values(need, "color", polarity=-1) == ["red"]
    assert _values(need, "color") == ["blue"]


def test_actual_negation_after_repair_cue_remains_a_remove():
    need, _ = _parse("red, actually not blue")
    assert _values(need, "color") == ["red"]
    assert _values(need, "color", polarity=-1) == ["blue"]


def test_repair_can_overwrite_a_normally_multi_value_material_slot():
    need, _ = _parse("leather before, but make it suede")
    assert _values(need, "material") == ["suede"]


def test_category_switch_and_following_use_case_are_positive():
    need, _ = _parse("a dress… no, shoes for running")
    assert need.category == "shoe"
    assert _values(need, "category") == ["shoe"]
    assert _values(need, "use_case") == ["running"]


def test_boot_to_running_correction_switches_category_and_clears_hiking():
    need = NeedModel()
    need.revise([
        Constraint("category", "boot", operation="SET"),
        Constraint("use_case", "hiking", operation="ADD"),
    ])
    need.revise(SlotFiller().parse("bro like running kind", turn=2))
    assert need.category == "shoe"
    assert _values(need, "category") == ["shoe"]
    assert _values(need, "use_case") == ["running"]
    assert "hiking" not in _values(need, "use_case")


def test_wrong_category_in_recommendation_complaint_is_not_a_new_preference():
    need = NeedModel()
    need.revise([
        Constraint("category", "boot", operation="SET"),
        Constraint("use_case", "hiking", operation="ADD"),
    ])
    need.revise(SlotFiller().parse("running does why are you giving me snow boots", turn=2))
    assert need.category == "shoe"
    assert _values(need, "category") == ["shoe"]
    assert _values(need, "use_case") == ["running"]
    assert "hiking" not in _values(need, "use_case")


def test_no_preference_clears_slot_and_becomes_boundary():
    need, _ = _parse("I do not care about color anymore")
    assert not [c for c in need.constraints if c.slot == "color"]
    assert need.no_preference == {"color"}


def test_common_no_preference_phrasings_are_recognized():
    for message in (
        "I don't have a preference for color",
        "no color preference",
        "color doesn't matter",
        "whatever color is fine",
    ):
        need, _ = _parse(message)
        assert need.no_preference == {"color"}, message


def test_no_additional_preference_does_not_clear_an_existing_value():
    need = NeedModel()
    need.revise([Constraint("color", "red", operation="SET")])
    need.revise(SlotFiller().parse(
        "I don't have an additional preference for color", turn=2,
    ))
    assert _values(need, "color") == ["red"]
    assert "color" not in need.no_preference


def test_keep_and_drop_produce_positive_and_negative_active_state():
    need, _ = _parse("keep waterproof, drop leather")
    assert _values(need, "feature") == ["waterproof"]
    assert _values(need, "material", polarity=-1) == ["leather"]


def test_remove_cue_rejects_an_exclusive_value():
    need, _ = _parse("remove red")
    assert _values(need, "color", polarity=-1) == ["red"]


def test_explicit_negation_wins_over_positive_match_in_same_turn():
    need, _ = _parse("like instead of linen i want polyester or something")
    assert _values(need, "material", polarity=-1) == ["linen"]
    assert _values(need, "material") == ["polyester"]
    assert "linen" not in [c.value for c in need.positives()]


def test_dont_want_does_not_emit_a_positive_value_or_spurious_feature():
    need, _ = _parse("i don't want linen, give me polyester")
    assert _values(need, "material", polarity=-1) == ["linen"]
    assert _values(need, "material") == ["polyester"]
    assert "want" not in [c.value for c in need.constraints]


def test_same_turn_llm_positive_cannot_resurrect_rejected_value():
    need = NeedModel()
    need.revise(SlotFiller().parse("instead of linen, polyester", turn=1))
    need.revise([Constraint("material", "linen", turn=1, operation="ADD")])
    assert _values(need, "material", polarity=-1) == ["linen"]
    assert "linen" not in _values(need, "material")


def test_rejected_value_masks_slot_and_tag_profile_preferences():
    from src.agent import Agent
    from src.context_engine import ProfilePreference, UserProfile

    need, _ = _parse("instead of linen, polyester")
    state = ConversationState(user_profile={"preference_tags": ["linen", "polyester"]})
    state.need = need
    state.profile = UserProfile("u", prefs=[
        ProfilePreference("material", "linen", 1.0, 0.0),
        ProfilePreference("tag", "linen", 1.0, 0.0),
    ])
    assert Agent._personalization_profile(state)["preference_tags"] == ["polyester"]


def test_effective_query_strictly_excludes_superseded_terms():
    message = (
        "i am looking for a fluffy slipper oh wait actly nah a bucket better like red color "
        "actually fuck lets do blue instead eh yellow better"
    )
    need, state = _parse(message)
    query_words = set(state.query_text().lower().split())
    assert need.category == "hat"
    assert _values(need, "color") == ["yellow"]
    assert {"slipper", "red", "blue"}.isdisjoint(query_words)
    assert {"hat", "yellow"}.issubset(query_words)


def test_effective_query_excludes_values_superseded_on_later_turns():
    filler = SlotFiller()
    state = ConversationState(user_profile={})
    state.accumulate("I want a red dress", turn=1)
    state.need.revise(filler.parse(state.all_text[-1], turn=1))
    state.accumulate("wait no, a blue shoe instead", turn=2)
    state.need.revise(filler.parse(state.all_text[-1], turn=2))

    query_words = set(state.query_text().lower().split())
    assert {"red", "dress"}.isdisjoint(query_words)
    assert {"blue", "shoe"}.issubset(query_words)


def test_query_is_structured_only_and_never_raw_transcript():
    need, state = _parse(
        "I want a cotton shirt... actually, make that linen. wait nah gimmie running shoes"
    )
    assert state.query_text() == "shoe running"
    assert "gimmie" not in state.query_text()


def test_leaky_query_switch_uses_raw_accumulated_history_only_when_marked():
    state = ConversationState(user_profile={})
    state.accumulate("I want a red dress", turn=1)
    state.need.revise(SlotFiller().parse(state.all_text[-1], turn=1))
    state.accumulate("actually make that a blue shoe", turn=2)
    state.need.revise(SlotFiller().parse(state.all_text[-1], turn=2))
    assert state.query_text() == "shoe blue"
    state.leaky_evidence = True
    assert state.query_text() == "I want a red dress actually make that a blue shoe"


def test_active_session_constraints_filter_conflicting_durable_profile_tags():
    from src.agent import Agent
    from src.context_engine import ProfilePreference, UserProfile

    need = NeedModel()
    need.revise([
        Constraint("category", "boot", operation="SET"),
        Constraint("use_case", "hiking", operation="ADD"),
    ])
    need.revise(SlotFiller().parse("running shoes", turn=2))
    state = ConversationState(user_profile={
        "preference_tags": ["boot", "hiking", "leather"], "summary": "",
    })
    state.need = need
    state.profile = UserProfile(
        "u", prefs=[
            ProfilePreference("category", "boot", 1.0, 0.0),
            ProfilePreference("use_case", "hiking", 1.0, 0.0),
            ProfilePreference("material", "leather", 1.0, 0.0),
        ],
    )
    projected = Agent._personalization_profile(state)
    assert projected["preference_tags"] == ["leather"]
    assert "boot" not in projected["preference_tags"]
    assert "hiking" not in projected["preference_tags"]


def test_profile_write_through_retires_superseded_durable_values():
    from src.context_engine import ProfileService, ProfilePreference, SessionContext, UserProfile
    from src.understanding import Belief

    need = NeedModel()
    need.revise([
        Constraint("category", "boot", operation="SET"),
        Constraint("use_case", "hiking", operation="ADD"),
    ])
    need.revise(SlotFiller().parse("running kind", turn=2))
    profile = UserProfile("u", prefs=[
        ProfilePreference("category", "boot", 1.0, 0.0),
        ProfilePreference("use_case", "hiking", 1.0, 0.0),
    ])
    ctx = SessionContext(need=need, belief=Belief())
    ProfileService(persistent=False).write_through(profile, ctx)
    assert {(pref.slot, pref.value) for pref in profile.prefs} == {
        ("category", "shoe"), ("use_case", "running"),
    }


def test_effective_constraint_phrases_remove_abandoned_values():
    need, state = _parse("red, actually blue")
    state.constraint_phrases = ["soft red waterproof", "lightweight"]
    assert state.effective_constraint_phrases() == ["soft waterproof", "lightweight"]


def test_size_tokens_do_not_leak_from_contractions_or_possessives():
    filler = SlotFiller()
    noisy = filler.parse("I'm looking for shoes for Valentine's Day", turn=1)
    assert not [event for event in noisy if event.slot == "size"]

    standalone = filler.parse("size m, then size 10", turn=1)
    assert [event.value for event in standalone if event.slot == "size"] == ["m", "size 10"]


def test_unparsed_active_sentence_terms_reach_effective_query():
    state = ConversationState(user_profile={})
    message = "I want something with a rich napped pile"
    state.accumulate(message, turn=1)
    state.need.revise(SlotFiller().parse(message, turn=1))
    query = state.query_text()
    assert {"rich", "napped", "pile"}.issubset(set(query.split()))
    assert "something" not in query.split()


def test_phrase_history_is_retired_on_repair():
    state = ConversationState(user_profile={})
    state.constraint_phrases = ["old wool preference", "new blue preference"]
    state.constraint_phrase_turns = [1, 2]
    state.invalidate_historical_phrases(2)
    assert state.constraint_phrases == ["new blue preference"]
    assert state.constraint_phrase_turns == [2]


def test_top_ten_requests_preserve_adaptive_reveal():
    from src.agent import Agent

    agent = object.__new__(Agent)
    agent.USE_ADAPTIVE_REVEAL = True
    agent.REVEAL_CONFIDENCE = 0.99
    agent.REVEAL_HOLDBACK_K = 1
    agent.REVEAL_TURN_CAP = 10
    agent.REVEAL_REQUIRE_CONSTRAINTS = False
    class State:
        belief = type("Belief", (), {"confidence": 0.0})()
    assert agent._reveal_count(State(), turn=1, top_k=10, new_constraints=False) == 1


def test_llm_slot_parser_uses_the_same_update_contract():
    from src.llm_inference import LLMSlotExtractor

    parsed = LLMSlotExtractor._parse(
        '[{"slot":"color","value":"","polarity":0,"operation":"NO_PREFERENCE"}]'
    )
    assert parsed == [{
        "slot": "color", "value": "", "polarity": 0, "operation": "NO_PREFERENCE",
    }]
