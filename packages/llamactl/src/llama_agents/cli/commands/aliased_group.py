"""Fully lifted from https://click.palletsprojects.com/en/stable/extending-click/"""

import click


class AliasedGroup(click.Group):
    """
    Implements a subclass of Group that accepts a prefix for a command.
    If there was a command called push, it would accept pus as an alias (so long as it was unique):
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        rv = super().get_command(ctx, cmd_name)

        if rv is not None:
            return rv

        matches = [x for x in self.list_commands(ctx) if x.startswith(cmd_name)]

        if not matches:
            return None

        if len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])

        ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str, click.Command, list[str]]:
        # always return the full command name
        cmd_name, cmd, args = super().resolve_command(ctx, args)
        assert cmd is not None
        full_name: str = (
            cmd.name
            if cmd.name
            else cmd_name
            if isinstance(cmd_name, str)
            else "<none>"  # shouldn't happen
        )
        return (full_name, cmd, args)
