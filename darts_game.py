class Game:
    def __init__(self, players, mode='501'):
        if len(players) == 0:
            raise ValueError('Must have at least one player')

        self.mode = mode
        self.playing = len(players)
        self.players = []
        self.game_over = False

        self.round = 1
        self.next_player_id = 0

        if mode == '501':
            target = 501
        elif mode == '301':
            target = 301
        else:
            raise ValueError(f'mode {mode} is not a valid mode')

        for player_name in players:
            self.players.append(Player(player_name, target))

        print(self)

    def __str__(self):
        return '\n'.join([f'Player: {player.name}\tTarget: {player.target}' for player in self.players])

    def score_points(self, points: int, multiplier: int, player_id: int = None):
        if player_id is None:
            player_id = self.next_player_id
        target_player = self.players[player_id]
        target_player.score(points, multiplier)
        if target_player.darts_remaining == 0:
            if player_id < len(self.players) - 1:
                self.next_player_id += 1
            else:
                self.round += 1
                self.next_player_id = 0

    def force_next_round(self):
        if not all(player.darts_remaining == 3 for player in self.players):
            self.round += 1
            self.next_player_id = 0
            for player in self.players:
                player.reset_darts_remaining()

    def get_json(self) -> dict:
        return {
            'mode': self.mode,
            'round': self.round,
            'players': [
                {
                    'name': p.name,
                    'target': p.target,
                    'darts_remaining': p.darts_remaining
                }
                for p in self.players
            ],
            'next_player': self.next_player_id,
        }


class Player:
    def __init__(self, name: str, target: int):
        self.name = name
        self.target = target
        self.darts_remaining = 3

    def __str__(self):
        return f'Player: {self.name}\tTarget: {self.target}'

    def score(self, score: int, multiplier: int):
        pts = score * multiplier
        if pts < self.target:
            if self.target - pts == 1:
                self._bust('Cannot have 1 remaining!')
            else:
                self._adjust_target(pts)
        elif pts == self.target:
            if multiplier == 2:
                self._adjust_target(pts)
            else:
                self._bust('Need to finish on a double!')
        else:
            self._bust('Cannot exceed target!')

    def reset_darts_remaining(self):
        self.darts_remaining = 3

    def _adjust_target(self, pts: int):
        self.target -= pts
        self.darts_remaining -= 1
        print(self)

    def _bust(self, reason: str):
        print('Bust: ' + reason)
        self.darts_remaining = 0
