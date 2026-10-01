from __future__ import annotations

import json

from world.director import VILLAGE_FIELDS, WORLD_FIELDS, Director
from world.goods import Goods
from world.llm import ModelUnreachable, NoJSONAnswer
from world.village import Village
from world.world import World, build_default_world


class FakeModel:
    """A model that answers with whatever was queued up for it."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def ask(self, system: str, user: str) -> str:
        self.prompts.append(user)
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)

    def ask_json(self, system: str, user: str) -> object:
        answer = self.ask(system, user)
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            raise NoJSONAnswer(user, answer) from None


def plan(
    *changes: dict,
    reply: str = "done",
    schedule: list | None = None,
    asks: bool = False,
) -> dict:
    return {
        "reply": reply,
        "asks": asks,
        "changes": list(changes),
        "schedule": schedule or [],
    }


def change(village: str | None, field: str, value: object) -> dict:
    return {"village": village, "field": field, "value": value}


class TestWhatMayBeChanged:
    def test_the_seed_is_nowhere_to_be_found(self) -> None:
        assert "seed" not in WORLD_FIELDS
        assert not any(name == "seed" for name in VILLAGE_FIELDS)

    def test_stock_may_be_changed_for_every_commodity(self) -> None:
        for goods in Goods:
            assert f"stock.{goods.label}" in VILLAGE_FIELDS

    def test_production_may_be_changed_for_every_commodity(self) -> None:
        for goods in Goods:
            assert f"production.{goods.label}" in VILLAGE_FIELDS

    def test_the_rules_of_the_run_may_be_changed(self) -> None:
        assert set(WORLD_FIELDS) == {"time_per_day", "trade_chance", "max_rounds"}

    def test_a_brain_may_not_be_swapped_out(self) -> None:
        assert not any("brain" in name for name in VILLAGE_FIELDS)

    def test_the_prompt_tells_the_model_the_seed_is_off_limits(self) -> None:
        world = build_default_world()
        model = FakeModel(plan(reply="nothing"))

        Director(model, world).follow("change the seed")

        assert "seed" in json.loads(model.prompts[0].split("\n\nThe user says")[0])[
            "not_changeable"
        ]


class TestApplyingNow:
    def test_a_village_setting_lands(self) -> None:
        world = build_default_world()

        Director(FakeModel(plan(change("Miners", "production.Minerals", 4.0))), world).follow(
            "half the mines"
        )

        assert world.village("Miners").production[Goods.MINERAL] == 4.0

    def test_a_world_setting_lands(self) -> None:
        world = build_default_world()

        Director(FakeModel(plan(change(None, "trade_chance", 0.9))), world).follow("talk more")

        assert world.trade_chance == 0.9

    def test_a_stock_setting_lands(self) -> None:
        world = build_default_world()

        Director(FakeModel(plan(change("Farmers", "stock.Agriculture", 0.0))), world).follow(
            "empty the farms"
        )

        assert world.village("Farmers").stock[Goods.FARM] == 0.0

    def test_a_whole_number_setting_lands_as_a_whole_number(self) -> None:
        world = build_default_world()

        Director(FakeModel(plan(change("Farmers", "population", 12.0))), world).follow("grow")

        village = world.village("Farmers")
        assert village.population == 12
        assert isinstance(village.population, int)

    def test_several_changes_all_land(self) -> None:
        world = build_default_world()
        answer = plan(
            change("Miners", "production.Minerals", 4.0),
            change(None, "trade_chance", 0.9),
        )

        Director(FakeModel(answer), world).follow("harsher")

        assert world.village("Miners").production[Goods.MINERAL] == 4.0
        assert world.trade_chance == 0.9

    def test_an_instruction_naming_nothing_changes_nothing(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(reply="I left it alone")), world).follow("carry on")

        assert reply.applied == []
        assert world.trade_chance == 0.6


class TestHoldingValues:
    def test_a_value_above_the_range_is_held_down_and_said_so(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.7))), world).follow("always")

        assert world.trade_chance == 1.0
        assert "held down to 1" in reply.line()

    def test_a_value_below_the_range_is_held_up_and_said_so(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Farmers", "stock.Agriculture", -50.0))), world).follow(
            "negative please"
        )

        assert world.village("Farmers").stock[Goods.FARM] == 0.0
        assert "held up to 0" in reply.line()

    def test_a_value_inside_the_range_is_left_alone_and_not_complained_about(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 0.8))), world).follow("a bit")

        assert "held" not in reply.line()

    def test_stock_is_held_at_zero_from_below(self) -> None:
        world = build_default_world()

        Director(FakeModel(plan(change("Farmers", "stock.Minerals", -1e9))), world).follow("negative")

        assert world.village("Farmers").stock[Goods.MINERAL] == 0.0

    def test_a_held_value_says_what_was_asked_for(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.7))), world).follow("always")

        assert reply.applied[0].asked == 1.7
        assert reply.applied[0].applied == 1.0

    def test_a_change_to_what_is_already_the_case_is_not_claimed_as_a_change(self) -> None:
        world = build_default_world()
        before = world.village("Miners").production[Goods.MINERAL]

        reply = Director(FakeModel(plan(change("Miners", "production.Minerals", before))), world).follow(
            "keep the mines as they are"
        )

        assert reply.applied == []
        assert "was already 20" in reply.line()

    def test_a_value_that_had_to_be_moved_is_not_reported_as_asked(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.7))), world).follow("always")

        assert "trade_chance = 1.7" not in reply.line()


class TestShowingWhatChanged:
    def test_a_change_shows_the_old_value_and_the_new(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Miners", "production.Minerals", 4.0))), world).follow(
            "half the mines"
        )

        assert reply.line().endswith("Miners production.Minerals: 20 -> 4")

    def test_a_world_change_shows_its_old_value_too(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 0.9))), world).follow("talk more")

        assert reply.line().endswith("trade_chance: 0.6 -> 0.9")

    def test_two_changes_to_one_setting_chain_rather_than_repeat(self) -> None:
        world = build_default_world()
        answer = plan(
            change("Miners", "production.Minerals", 10.0),
            change("Miners", "production.Minerals", 5.0),
        )

        reply = Director(FakeModel(answer), world).follow("halve it twice")

        assert [
            line for line in reply.line().splitlines() if line.startswith("Miners")
        ] == [
            "Miners production.Minerals: 20 -> 10",
            "Miners production.Minerals: 10 -> 5",
        ]

    def test_a_held_value_still_shows_both_ends(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.7))), world).follow("always")

        assert "trade_chance: 0.6 -> 1" in reply.line()

    def test_the_written_record_says_what_was_asked_for(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.7))), world).follow("always")

        assert "[asked for 1.7]" in reply.applied[0].log_line()

    def test_the_written_record_says_nothing_extra_when_nothing_was_held(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 0.9))), world).follow("talk more")

        assert reply.applied[0].log_line() == reply.applied[0].line()

    def test_a_change_held_for_later_is_not_shown_as_a_value_changing(self) -> None:
        world = build_default_world()
        answer = plan(
            schedule=[
                {
                    "day": 7,
                    "label": "the drought",
                    "changes": [change("Farmers", "production.Agriculture", 2.0)],
                }
            ]
        )

        reply = Director(FakeModel(answer), world).follow("a drought on day 7")

        assert reply.applied == []
        assert reply.held == ["held for day 7, the drought: production.Agriculture"]

    def test_a_held_change_reads_the_old_value_when_it_lands(self) -> None:
        """The old value is whatever it was on the day, not the day it was planned."""
        world = build_default_world()
        answer = plan(
            schedule=[
                {
                    "day": 2,
                    "label": "the drought",
                    "changes": [change("Farmers", "production.Agriculture", 2.0)],
                }
            ]
        )
        reply = Director(FakeModel(answer), world).follow("drought")
        world.advance_day()
        world.advance_day()

        assert reply.held
        assert world.village("Farmers").production[Goods.FARM] == 2.0

    def test_numbers_are_written_without_a_pointless_decimal(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "trade_chance", 1.0))), world).follow("always")

        assert reply.line().endswith("trade_chance: 0.6 -> 1")


class TestRefusingWhatShouldNotHappen:
    def test_the_seed_is_refused_wherever_it_is_aimed(self) -> None:
        world = build_default_world()
        answer = plan(
            change("Artisans", "seed", 7),
            change(None, "seed", 9),
        )

        reply = Director(FakeModel(answer), world).follow("reroll")

        assert reply.applied == []
        assert world.seed == 0
        assert world.village("Artisans").seed == 2
        assert reply.line().count("seed") == 2

    def test_an_invented_field_is_refused(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Artisans", "wing_size", 3))), world).follow("wings")

        assert reply.applied == []
        assert "wing_size" in reply.line()

    def test_a_village_that_does_not_exist_is_refused(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Ferrymen", "population", 5))), world).follow("more")

        assert reply.applied == []
        assert "Ferrymen" in reply.line()

    def test_a_value_that_is_not_a_number_is_refused(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Miners", "population", "lots"))), world).follow("grow")

        assert reply.applied == []
        assert "rather than a number" in reply.line()

    def test_a_value_that_is_a_boolean_is_refused(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "time_per_day", True))), world).follow("more")

        assert reply.applied == []

    def test_a_change_that_is_not_an_object_is_refused(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel({"reply": "x", "changes": ["nonsense"]}), world).follow("go")

        assert reply.applied == []
        assert "not an object" in reply.line()

    def test_a_whole_setting_left_alone_is_not_the_world_setting(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Miners", "trade_chance", 0.2))), world).follow("go")

        assert reply.applied == []
        assert world.trade_chance == 0.6

    def test_the_good_changes_land_even_when_a_bad_one_is_refused(self) -> None:
        world = build_default_world()
        answer = plan(
            change("Miners", "production.Minerals", 4.0),
            change("Artisans", "seed", 7),
        )

        reply = Director(FakeModel(answer), world).follow("half the mines")

        assert world.village("Miners").production[Goods.MINERAL] == 4.0
        assert len(reply.applied) == 1
        assert len(reply.refused) == 1


    def test_naming_a_village_for_a_world_setting_says_what_to_do(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Miners", "trade_chance", 0.2))), world).follow("go")

        assert "setting of the world" in reply.line()
        assert "must not name a village" in reply.line()

    def test_naming_the_world_for_a_village_setting_says_what_to_do(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change(None, "population", 20))), world).follow("go")

        assert "setting of one village" in reply.line()

    def test_an_invented_field_lists_what_is_changeable(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Artisans", "wing_size", 4))), world).follow("go")

        assert "Changeable here" in reply.line()
        assert "production_noise" in reply.line()

    def test_an_unknown_village_lists_the_villages_that_are_there(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Ferrymen", "population", 4))), world).follow("go")

        assert "The villages are: Farmers, Miners, Artisans" in reply.line()


class TestHoldingChangesForALaterDay:
    def test_a_change_can_be_held_for_a_named_day(self) -> None:
        world = build_default_world()
        answer = plan(
            reply="a drought",
            schedule=[
                {
                    "day": 12,
                    "label": "the drought",
                    "changes": [change("Farmers", "production.Agriculture", 2.0)],
                }
            ],
        )

        Director(FakeModel(answer), world).follow("a drought on day 12")

        assert world.village("Farmers").production[Goods.FARM] == 20.0
        assert [(e.day, e.label) for e in world.scheduled] == [(12, "the drought")]

    def test_a_held_change_lands_on_the_day_it_was_promised_for(self) -> None:
        world = build_default_world()
        answer = plan(
            schedule=[
                {
                    "day": 3,
                    "label": "the drought",
                    "changes": [change("Farmers", "production.Agriculture", 2.0)],
                }
            ]
        )

        Director(FakeModel(answer), world).follow("drought on day 3")
        for _ in range(3):
            world.advance_day()

        assert world.village("Farmers").production[Goods.FARM] == 2.0

    def test_a_held_change_is_said_so_rather_than_left_silent(self) -> None:
        world = build_default_world()
        answer = plan(
            schedule=[{"day": 12, "label": "the drought", "changes": [change(None, "max_rounds", 2)]}]
        )

        reply = Director(FakeModel(answer), world).follow("fewer rounds later")

        assert "day 12" in reply.line()
        assert "the drought" in reply.line()

    def test_a_schedule_with_no_usable_day_is_dropped(self) -> None:
        world = build_default_world()
        answer = plan(
            schedule=[
                {"day": 0, "label": "never", "changes": [change(None, "max_rounds", 2)]},
                {"label": "no day", "changes": [change(None, "max_rounds", 2)]},
                {"day": "soon", "label": "word", "changes": [change(None, "max_rounds", 2)]},
            ]
        )

        reply = Director(FakeModel(answer), world).follow("later")

        assert world.scheduled == []
        assert reply.line().count("no usable day") == 3

    def test_a_schedule_with_nothing_in_it_is_dropped_quietly(self) -> None:
        world = build_default_world()
        answer = plan(schedule=[{"day": 5, "label": "nothing", "changes": []}])

        Director(FakeModel(answer), world).follow("later")

        assert world.scheduled == []

    def test_a_held_change_is_validated_like_any_other(self) -> None:
        world = build_default_world()
        answer = plan(
            schedule=[
                {
                    "day": 2,
                    "label": "impossible",
                    "changes": [change(None, "trade_chance", 99)],
                }
            ]
        )

        Director(FakeModel(answer), world).follow("later")
        world.advance_day()
        world.advance_day()

        assert world.trade_chance == 1.0

    def test_a_schedule_that_is_not_an_object_is_dropped(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel({"reply": "x", "changes": [], "schedule": ["soon"]}), world).follow(
            "later"
        )

        assert world.scheduled == []
        assert "not an object" in reply.line()


class TestABrokenAnswer:
    def test_prose_comes_back_with_the_instruction_and_an_ask_to_repeat(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel("I would lower the trade chance."), world).follow("harsher")

        assert "harsher" in reply.line()
        assert "I would lower the trade chance." in reply.line()
        assert "Say it again." in reply.line()

    def test_a_broken_answer_changes_nothing(self) -> None:
        world = build_default_world()

        Director(FakeModel("just prose"), world).follow("harsher")

        assert world.trade_chance == 0.6

    def test_the_whole_prompt_is_kept_for_the_log(self) -> None:
        world = build_default_world()
        model = FakeModel("just prose")

        reply = Director(model, world).follow("harsher")

        assert reply.prompt == model.prompts[0]
        assert "harsher" in reply.prompt

    def test_an_answer_that_is_not_an_object_asks_again(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel([1, 2, 3]), world).follow("harsher")

        assert reply.broke is not None
        assert "Say it again." in reply.line()

    def test_an_unreachable_model_is_reported_plainly(self) -> None:
        class Broken:
            def ask(self, system, user):
                raise ModelUnreachable("connection refused")

            def ask_json(self, system, user):
                raise ModelUnreachable("connection refused")

        world = build_default_world()

        reply = Director(Broken(), world).follow("harsher")

        assert "could not be reached" in reply.line()
        assert world.trade_chance == 0.6


class TestRefusingRatherThanGuessing:
    def test_a_reply_that_asks_changes_nothing(self) -> None:
        """Asking and doing in one turn leaves a reader unable to tell which."""
        world = build_default_world()
        asked = plan(
            change(None, "trade_chance", 0.0),
            reply="They stop meeting, I assume?",
            asks=True,
        )

        reply = Director(FakeModel(asked), world).follow("stop the villages meeting")

        assert reply.applied == []
        assert world.trade_chance == 0.6
        assert "question" in "\n".join(reply.refused)

    def test_a_rhetorical_question_is_not_an_order_to_stop(self) -> None:
        """A raised eyebrow is not a question, and must not leave an order undone."""
        world = build_default_world()
        rhetorically = plan(
            change("Farmers", "population", 20),
            reply="Twenty people for the Farmers? That will lift them.",
        )

        reply = Director(FakeModel(rhetorically), world).follow("give the Farmers twenty people")

        assert len(reply.applied) == 1
        assert world.village("Farmers").population == 20

    def test_an_order_still_goes_through_when_nothing_asked(self) -> None:
        world = build_default_world()
        ordered = plan(change(None, "trade_chance", 0.0), reply="They stop meeting.")

        reply = Director(FakeModel(ordered), world).follow("stop the villages meeting")

        assert len(reply.applied) == 1
        assert world.trade_chance == 0.0

    def test_a_reply_must_not_mention_a_simulator(self) -> None:
        """The people in this world do not know they are in one."""
        assert "simulation" not in Director.SYSTEM_PROMPT.lower()
        assert "simulator" not in Director.SYSTEM_PROMPT.lower()

    def test_a_part_the_model_cannot_express_is_shown_back(self) -> None:
        world = build_default_world()
        answer = {
            "reply": "There is no tax rate in this world.",
            "changes": [],
            "schedule": [],
            "unmapped": ["a tax rate for the Miners"],
        }

        reply = Director(FakeModel(answer), world).follow("a tax rate of ten percent")

        assert "tax rate" in reply.line()
        assert "Not done:" in reply.line()

    def test_naming_what_it_could_not_do_changes_nothing(self) -> None:
        world = build_default_world()
        answer = {"reply": "no", "changes": [], "schedule": [], "unmapped": ["a morale"]}

        Director(FakeModel(answer), world).follow("morale of 3")

        assert world.trade_chance == 0.6
        assert world.village("Farmers").production[Goods.FARM] == 20.0

    def test_the_prompt_forbids_approximating(self) -> None:
        prompt = Director.SYSTEM_PROMPT.lower()

        assert "never approximate" in prompt
        assert "unmapped" in prompt

    def test_the_prompt_gives_an_example_of_refusing(self) -> None:
        assert "tax rate" in Director.SYSTEM_PROMPT.lower()

    def test_a_refusal_is_shown_under_a_heading_not_as_narration(self) -> None:
        world = build_default_world()
        answer = plan(change("Artisans", "morale", 3), reply="I would if I could")

        lines = Director(FakeModel(answer), world).follow("morale of 3").line().splitlines()

        assert lines[0] == "I would if I could"
        assert "Not done:" in lines
        assert lines.index("Not done:") > 0

    def test_the_same_refusal_is_said_once(self) -> None:
        world = build_default_world()
        answer = plan(
            change("Quarries", "population", 5),
            change("Quarries", "production.Agriculture", 5),
            change("Quarries", "stock.Agriculture", 5),
        )

        reply = Director(FakeModel(answer), world).follow("build a fourth village")

        assert reply.line().count("Quarries") == 1

    def test_a_refusal_says_what_could_be_set_instead(self) -> None:
        world = build_default_world()

        reply = Director(FakeModel(plan(change("Artisans", "morale", 3))), world).follow("morale")

        assert "Changeable here" in reply.line()
        assert "population" in reply.line()

    def test_a_change_of_nothing_is_not_claimed_as_a_change(self) -> None:
        world = build_default_world()
        reply = Director(
            FakeModel(plan(change("Miners", "production.Minerals", 20.0))), world
        ).follow("keep the mines")

        assert reply.applied == []
        assert "nothing changed" in reply.line()


class TestWhatTheModelIsTold:
    def test_the_prompt_carries_the_current_state(self) -> None:
        world = build_default_world()
        model = FakeModel(plan(reply="x"))

        Director(model, world).follow("do something")

        sent = json.loads(model.prompts[0].split("\n\nThe user says")[0])
        assert sent["state"]["trade_chance"] == 0.6
        assert sent["state"]["villages"]["Miners"]["production"]["Minerals"] == 20.0

    def test_the_prompt_lists_what_may_be_changed(self) -> None:
        world = build_default_world()
        model = FakeModel(plan(reply="x"))

        Director(model, world).follow("do something")

        sent = json.loads(model.prompts[0].split("\n\nThe user says")[0])
        assert "population" in sent["village"]
        assert "trade_chance" in sent["world"]
        assert sent["commodities"] == ["Agriculture", "Minerals", "Handicrafts"]

    def test_the_prompt_carries_the_ranges(self) -> None:
        world = build_default_world()
        model = FakeModel(plan(reply="x"))

        Director(model, world).follow("do something")

        sent = json.loads(model.prompts[0].split("\n\nThe user says")[0])
        assert sent["ranges"]["trade_chance"] == [0.0, 1.0]

    def test_the_prompt_names_the_villages_that_are_there(self) -> None:
        world = World([Village("Only", 10, {g: 5.0 for g in Goods}, {g: 1.0 for g in Goods})])
        model = FakeModel(plan(reply="x"))

        Director(model, world).follow("do something")

        sent = json.loads(model.prompts[0].split("\n\nThe user says")[0])
        assert sent["villages"] == ["Only"]

    def test_the_prompt_says_which_day_is_next(self) -> None:
        world = build_default_world()
        world.advance_day()
        model = FakeModel(plan(reply="x"))

        Director(model, world).follow("do something")

        sent = json.loads(model.prompts[0].split("\n\nThe user says")[0])
        assert sent["today"] == 2
        assert sent["state"]["day"] == 1