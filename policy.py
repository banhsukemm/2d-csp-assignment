import random
from abc import abstractmethod

import numpy as np


class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Random choice a stock idx
                pos_x, pos_y = None, None
                for _ in range(100):
                    # random choice a stock
                    stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                    stock = observation["stocks"][stock_idx]

                    # Random choice a position
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x = random.randint(0, stock_w - prod_w)
                        pos_y = random.randint(0, stock_h - prod_h)
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x = random.randint(0, stock_w - prod_h)
                        pos_y = random.randint(0, stock_h - prod_w)
                        if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                            prod_size = prod_size[::-1]
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
class RandomPolicy2(Policy):
    def __init__(self):
        self.cur_stock_idx = -1
        self.solution = []
        self.loop = 100
        self.stop_condition = 0.05

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0.0:
            self.cur_stock_idx = -1
            self.solution = []

        # Sắp xếp các stock theo kích thước giảm dần
        sorted_stock = sorted(
            enumerate(observation["stocks"]),
            key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
            reverse=True,
        )

        if self.solution:
            return self.solution.pop(0)
        else:
            self.cur_stock_idx += 1
            idx, stock = sorted_stock[self.cur_stock_idx]
            self.solution = self.find_solution(stock.copy(), idx, observation["products"])
            return self.solution.pop(0)

    def calculate_trim_loss(self, c_stock):
        total_area = np.sum(c_stock >= -1)
        trim_loss = np.sum(c_stock == -1)
        return trim_loss / total_area if total_area > 0 else 0.0

    def find_solution(self, stock, stock_idx, list_prod):
        best_solution = []
        best_trim_loss = self.calculate_trim_loss(stock)

        for _ in range(self.loop):

            c_list_prod = [prod.copy() for prod in list_prod]
            c_stock = stock.copy()
            new_solution = []

            while True:
                has_placement = False
                for __ in range(100):
                    prod_idx = random.randint(0, len(c_list_prod) - 1)
                    prod = c_list_prod[prod_idx]
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        stock_w, stock_h = self._get_stock_size_(c_stock)
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            for _ in range(10):
                                pos_x = random.randint(0, stock_w - prod_w)
                                pos_y = random.randint(0, stock_h - prod_h)
                                if self._can_place_(c_stock, (pos_x, pos_y), prod_size):
                                    new_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
                                    c_list_prod[prod_idx]["quantity"] -= 1
                                    c_stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 0
                                    has_placement = True
                                    break
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for _ in range(10):
                                pos_x = random.randint(0, stock_w - prod_h)
                                pos_y = random.randint(0, stock_h - prod_w)
                                if self._can_place_(c_stock, (pos_x, pos_y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    new_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
                                    c_list_prod[prod_idx]["quantity"] -= 1
                                    c_stock[pos_x : pos_x + prod_h, pos_y : pos_y + prod_w] = 0
                                    has_placement = True
                                    break

                        if has_placement:
                            break

                if not has_placement:  
                    break

            new_trim_loss = self.calculate_trim_loss(c_stock)
            if new_trim_loss < best_trim_loss:
                best_solution = new_solution
                best_trim_loss = new_trim_loss
            if best_trim_loss < self.stop_condition:
                break

        return best_solution



class GreedyPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class GreedyPolicy2(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        

                # Loop through all stocks
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            pos_x, pos_y = None, None
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

            if pos_x is not None and pos_y is not None:
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

