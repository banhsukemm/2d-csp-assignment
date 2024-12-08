from policy import Policy
import numpy as np

class Policy2314047(Policy):
    def __init__(self):
        self.sorted_stock = None
        self.sorted_prod = None
        self.cur_stock_idx = -1
        self.cur_prod_idx = -1

    def get_action(self, observation, info):
        sorted_prod = sorted(enumerate(observation["products"]), 
                                key=lambda p: p[1]["size"][0] * p[1]["size"][1], 
                                reverse=True)
            
        sorted_stock = sorted(enumerate(observation["stocks"]),
                                   key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
                                   reverse=True)
        if self.cur_stock_idx != -1 and self.cur_prod_idx != -1:
            stock = observation["stocks"][self.cur_stock_idx]
            prod = observation["products"][self.cur_prod_idx]
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size
                if (prod_w <= stock_w and prod_h <= stock_h):
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": self.cur_stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)} 
                if (prod_w <= stock_h and prod_h <= stock_w):
                    prod_size = [prod_h, prod_w]
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": self.cur_stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)}

                
        
        for idx, stock in sorted_stock:
            stock_w, stock_h = self._get_stock_size_(stock)

            for pidx, prod in sorted_prod:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    
                    if (prod_w <= stock_w and prod_h <= stock_h):
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    self.cur_prod_idx = pidx
                                    self.cur_stock_idx = idx
                                    return {"stock_idx": idx,
                                            "size": prod_size,
                                            "position": (x, y)}     
                    if (prod_w <= stock_h and prod_h <= stock_w):
                        prod_size = [prod_h, prod_w]
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    self.cur_prod_idx = pidx
                                    self.cur_stock_idx = idx
                                    return {"stock_idx": idx,
                                            "size": prod_size,
                                            "position": (x, y)}

        self.cur_stock_idx = -1
        self.cur_prod_idx = -1
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.list_prod = None
            self.sorted_stock = None
            self.sorted_prod = None
            self.cur_prod = None
            self.cur_stock = None
            self.cur_idx = -1
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0


            if self.list_prod is None and not np.array_equal(self.list_prod, observation["products"]):
                print("New Env")
                self.cur_prod = None
                self.cur_stock = None
                self.cur_idx = -1

                self.sorted_prod = sorted(observation["products"], 
                                    key=lambda p: p["size"][0] * p["size"][1], 
                                    reverse=True)
                
            self.sorted_stock = sorted(enumerate(observation["stocks"]),
                                    key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
                                    reverse=True)
            
            if self.cur_stock is not None and self.cur_prod is not None and self.cur_prod["quantity"] > 0:
                stock_w, stock_h = self._get_stock_size_(self.cur_stock)
                prod_size = self.cur_prod["size"]
                prod_w, prod_h = prod_size
                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(self.cur_stock, (x, y), prod_size):
                                print("No rotate")
                                return {"stock_idx": self.cur_idx, 
                                        "size": prod_size, 
                                        "position": (x, y)}
                elif stock_w >= prod_h and stock_h >= prod_w:
                    prod_size = [prod_h, prod_w]
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(self.cur_stock, (x, y), prod_size):
                                print("Rotated")
                                return {"stock_idx": self.cur_idx, 
                                        "size": prod_size, 
                                        "position": (x, y)}

            for idx, stock in self.sorted_stock:
                if idx < self.cur_idx:
                    continue

                stock_w, stock_h = self._get_stock_size_(stock)
                pos_x, pos_y = None, None

                for prod in self.sorted_prod:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size
                        
                        if (prod_w <= stock_w and prod_h <= stock_h):
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        print("No rotate")
                                        self.cur_stock = stock
                                        self.cur_idx = idx
                                        self.cur_prod = prod
                                        return {"stock_idx": idx,
                                                "size": prod_size,
                                                "position": (x, y)}
                        elif (prod_w <= stock_h and prod_h <= stock_w):
                            prod_size = [prod_h, prod_w]
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        print("No rotate")
                                        self.cur_stock = stock
                                        self.cur_idx = idx
                                        self.cur_prod = prod
                                        return {"stock_idx": idx,
                                                "size": prod_size,
                                                "position": (x, y)}           

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        elif self.policy_id == 2:
            pass

    # Student code here
    # You can add more functions if needed