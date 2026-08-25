import random
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # p1 plays first
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str([item for sub in self.board for item in sub])
        return self.boardHash

    def winner(self):
        # rows
        for i in range(BOARD_ROWS):
            s = sum(self.board[i])
            if s == 3:
                self.isEnd = True
                return 1
            if s == -3:
                self.isEnd = True
                return -1

        # cols
        for i in range(BOARD_COLS):
            s = sum([self.board[j][i] for j in range(BOARD_ROWS)])
            if s == 3:
                self.isEnd = True
                return 1
            if s == -3:
                self.isEnd = True
                return -1

        # diagonals
        d1 = sum([self.board[i][i] for i in range(BOARD_ROWS)])
        d2 = sum([self.board[i][BOARD_ROWS - 1 - i] for i in range(BOARD_ROWS)])
        if d1 == 3 or d2 == 3:
            self.isEnd = True
            return 1
        if d1 == -3 or d2 == -3:
            self.isEnd = True
            return -1

        # tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position[0]][position[1]] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        wins, losses, draws = 0, 0, 0
        for _ in range(rounds):
            while not self.isEnd:
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    if win == 1:
                        wins += 1
                    elif win == -1:
                        losses += 1
                    else:
                        draws += 1
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                board_hash = self.getHash()
                self.p2.addState(board_hash)

                win = self.winner()
                if win is not None:
                    if win == 1:
                        wins += 1
                    elif win == -1:
                        losses += 1
                    else:
                        draws += 1
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
        return wins, losses, draws

    def play_human(self):
        while not self.isEnd:
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                self.announce(win)
                self.reset()
                break

            positions = self.availablePositions()
            p2_action = self.p2.chooseAction(positions)
            self.updateState(p2_action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                self.announce(win)
                self.reset()
                break

    def announce(self, win):
        if win == 1:
            print(f"{self.p1.name} wins!")
        elif win == -1:
            print(f"{self.p2.name} wins!")
        else:
            print("It's a draw!")

    def showBoard(self):
        symbols = {1: "X", -1: "O", 0: " "}
        print()
        for i in range(BOARD_ROWS):
            print("| " + " | ".join(symbols[self.board[i][j]] for j in range(BOARD_COLS)) + " |")
        print()


class Player:
    """RL Agent: uses a state-value function, updated via TD learning."""

    def __init__(self, name, exp_rate=0.05, lr=0.5):
        self.name = name
        self.states = []
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        return str([item for sub in board for item in sub])

    def chooseAction(self, positions, current_board, symbol):
        if random.uniform(0, 1) <= self.exp_rate:
            idx = random.choice(range(len(positions)))
            return positions[idx]

        value_max = -999
        action = positions[0]
        for p in positions:
            next_board = [row[:] for row in current_board]
            next_board[p[0]][p[1]] = symbol
            next_hash = self.getHash(next_board)
            value = self.states_value.get(next_hash, 0)
            if value >= value_max:
                value_max = value
                action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self, filename="policy.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, filename="policy.pkl"):
        with open(filename, "rb") as f:
            self.states_value = pickle.load(f)


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input row (0-2): "))
            col = int(input("Input column (0-2): "))
            if (row, col) in positions:
                return (row, col)
            print("Invalid move, try again.")

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass


def train_and_report(episodes, explore_p1=0.05, lr_p1=0.5, explore_p2=0.05, lr_p2=0.5):
    p1 = Player("Agent1", exp_rate=explore_p1, lr=lr_p1)
    p2 = Player("Agent2", exp_rate=explore_p2, lr=lr_p2)
    st = State(p1, p2)

    print(f"\nTraining for {episodes} episodes...")
    wins, losses, draws = st.play(rounds=episodes)
    total = wins + losses + draws
    print(f"Win %:  {wins / total * 100:.1f}%")
    print(f"Loss %: {losses / total * 100:.1f}%")
    print(f"Draw %: {draws / total * 100:.1f}%")
    return p1, p2


if __name__ == "__main__":
    # Train the agent - change episode counts to reproduce the table
    # (100, 500, 1000, 5000, 10000)
    trained_p1, trained_p2 = train_and_report(
        episodes=1000,
        explore_p1=0.05,
        lr_p1=0.5,
        explore_p2=0.05,
        lr_p2=0.5,
    )

    # Save trained policy
    trained_p1.savePolicy("policy_p1.pkl")

    # Play against the trained agent
    play = input("\nPlay against the trained agent? (y/n): ")
    if play.lower() == "y":
        trained_p1.exp_rate = 0
        human = HumanPlayer("You")
        game = State(trained_p1, human)
        game.play_human()
