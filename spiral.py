"""

Spiral sensor class

Exposed methods:
    - draw_spiral()
    - analyze()

"""


# Standard library packages
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, integrate
from typing import Callable

# Package modules
from .models import Models


class Spiral:
    """
    Spiral class

    Archimedes spiral: "r(theta) = a + b*theta", where:
        - "a": outer spiral starting point offset from the origin along the "x" axis
        - "2*pi*b": distance between the line pairs in the spiral.
        - theta: angle from the "x" axis
        - the outer and inner spiral waveguides are joined by an S-bend at the center

        In this case, the center-to-center spacing between the rings in the spiral,
        i.e. the "linewidth", is 2 x (waveguide core width + inter-waveguide spacing),
        thus "2*pi*b" = 2 x (waveguide core width + inter-waveguide spacing).

        - Constraints by hypothesis (MAGIC NUMBERS):
            1) min{a} = 1.5 x linewidth to allow for space for the S-bend joint at the
               center of the spiral.
            2) The spiral must have at least one complete turn, else it's just a
               "curved waveguides", therefore the minimum spiral radius = a + linewidth,
               but this parameter us user-selectable in the .toml file.

    All lengths are in units of um
    """

    def __init__(
        self,
        spacing: float,
        turns_min: float,
        turns_max: float,
        models: Models,
        logger: Callable = print,
    ):

        # Load class instance parameter values
        self.spacing: float = spacing
        self.turns_min: float = turns_min
        self.turns_max: float = turns_max
        self.models: Models = models
        self.logger: Callable = logger

        # Calculate spiral parameters
        self.line_width: float = 2 * (self.models.core_v_value + self.spacing)
        self.a_spiral: float = 1.5 * self.line_width
        self.b_spiral: float = self.line_width / (2 * np.pi)

        # Declare class instance internal variables
        self.previous_solution: np.ndarray = np.asarray([-1, -1])

        # Declare class instance result variables and arrays
        self.S: np.ndarray = np.ndarray([])
        self.h: np.ndarray = np.ndarray([])
        self.n_turns: np.ndarray = np.ndarray([])
        self.outer_spiral_r_min: np.ndarray = np.ndarray([])
        self.L: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.max_S: float = 0
        self.max_S_radius: float = 0
        self.results: list = []

    def draw_spiral(
        self,
        r_outer: float,
        h: float,
        n_turns: float,
        r_window: float,
        figure: plt.Figure = None,
    ) -> plt.Figure:
        """
        This function actually draws a rotated version of the spiral, where the inside
        starting point is always on the x axis, i.e. theta_min = 0.
        """

        # Sensitivity
        S = self._calc_sensitivity(h=h, n_turns=n_turns, r=r_outer)[0]

        # Archimedes spiral parameters
        a_spiral: float = r_outer - self.line_width * n_turns
        L: float = integrate.quad(func=self._line_element, a=a_spiral, b=r_outer)[0]
        theta_max: float = (r_outer - a_spiral) / self.b_spiral
        thetas_spiral: np.ndarray = np.linspace(0, theta_max, 1000)
        r_spiral: np.ndarray = a_spiral + (self.b_spiral * thetas_spiral)

        # Half circle and S-bend joint parameters
        thetas_joint: np.ndarray = -np.linspace(0, np.pi, 100)
        r_half_circle: np.ndarray = a_spiral + (self.b_spiral * thetas_joint)
        r_S_bend: float = r_half_circle[-1] + self.models.core_v_value

        # Define new figure if required, else use axes passed as a function parameter
        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.axes[0]
            ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-r_window, r_window)
        ax.set_ylim(-r_window, r_window)
        ax.set_title(
            "Archimedes spiral : "
            + f"{n_turns: .2f} turns, "
            + "".join([f"w = {self.models.core_v_value:.3f} ", r"$\mu$m, "])
            + "".join([f"spacing = {self.spacing:.1f} ", r"$\mu$m, "])
            + "".join([f"h = {h:.3f} ", r"$\mu$m, "])
            + "".join([f"S = {S:.0f}", r" RIU$^{-1}$"])
            + "\n"
            + "".join([f"R = {r_outer:.1f} ", r"$\mu$m, "])
            + "".join([r"R$_{min}$ = ", f"{a_spiral:.1f} ", r"$\mu$m, "])
            + "".join([r"S-bend radius = ", f"{a_spiral/2:.1f} ", r"$\mu$m, "])
            + "".join([f"L = {L:.1f} ", r"$\mu$m"])
        )
        ax.set_xlabel(r"$\mu$m")
        ax.set_ylabel(r"$\mu$m")

        # Inner spiral
        ax.plot(
            r_spiral * np.cos(thetas_spiral),
            r_spiral * np.sin(thetas_spiral),
            color="blue",
        )
        ax.plot(
            (r_spiral + self.models.core_v_value) * np.cos(thetas_spiral),
            (r_spiral + self.models.core_v_value) * np.sin(thetas_spiral),
            color="blue",
        )

        # Outer spiral
        ax.plot(
            (r_spiral + self.models.core_v_value + self.spacing)
            * np.cos(thetas_spiral),
            (r_spiral + self.models.core_v_value + self.spacing)
            * np.sin(thetas_spiral),
            color="red",
        )
        ax.plot(
            (r_spiral + 2 * self.models.core_v_value + self.spacing)
            * np.cos(thetas_spiral),
            (r_spiral + 2 * self.models.core_v_value + self.spacing)
            * np.sin(thetas_spiral),
            color="red",
        )

        # Joint: outer waveguide half circle
        ax.plot(
            (r_half_circle + self.models.core_v_value + self.spacing)
            * np.cos(thetas_joint),
            (r_half_circle + self.models.core_v_value + self.spacing)
            * np.sin(thetas_joint),
            color="red",
        )
        ax.plot(
            (r_half_circle + 2 * self.models.core_v_value + self.spacing)
            * np.cos(thetas_joint),
            (r_half_circle + 2 * self.models.core_v_value + self.spacing)
            * np.sin(thetas_joint),
            color="red",
        )

        # Joint: inner waveguide "S-bend"
        ax.plot(
            (a_spiral + self.models.core_v_value / 2) / 2
            + (a_spiral - self.models.core_v_value / 2) / 2 * np.cos(thetas_joint),
            (a_spiral - self.models.core_v_value / 2) / 2 * np.sin(thetas_joint),
            color="blue",
        )
        ax.plot(
            (a_spiral + self.models.core_v_value / 2) / 2
            + (a_spiral + 1.5 * self.models.core_v_value) / 2 * np.cos(thetas_joint),
            (a_spiral + 1.5 * self.models.core_v_value) / 2 * np.sin(thetas_joint),
            color="blue",
        )
        ax.plot(
            -(
                (r_S_bend + self.models.core_v_value / 2 + self.spacing) / 2
                + (r_S_bend - self.models.core_v_value / 2 + self.spacing)
                / 2
                * np.cos(thetas_joint)
            ),
            -(r_S_bend - self.models.core_v_value / 2 + self.spacing)
            / 2
            * np.sin(thetas_joint),
            color="blue",
        )
        ax.plot(
            -(
                (r_S_bend + self.models.core_v_value / 2 + self.spacing) / 2
                + (r_S_bend + 3 * self.models.core_v_value / 2 + self.spacing)
                / 2
                * np.cos(thetas_joint)
            ),
            -(r_S_bend + 3 * self.models.core_v_value / 2 + self.spacing)
            / 2
            * np.sin(thetas_joint),
            color="blue",
        )

        return fig

    def _line_element(self, r: float) -> float:
        """
        Line element for numerical integration in polar coordinates (r, theta)
        "dl = sqrt(r**2 + (dr/dtheta)**2)*dtheta", converted to a function of "r" only
        with the spiral equation (r = a + b*theta) to "dl(r) = sqrt((r/b)**2 + 1)*dr".
        """
        return np.sqrt((r / self.b_spiral) ** 2 + 1)

    def _line_element_bend_loss(self, r: float, h: float) -> float:
        """
        Bending losses for a line element: alpha_bend(r)*dl(r)
        """

        return self.models.alpha_bend(r=r, u=h) * self._line_element(r=r)

    def _calc_sensitivity(
        self, r: float, h: float, n_turns: float
    ) -> tuple[float, float, float]:
        """
        Calculate sensitivity at radius r for a given core height and number of turns
        """

        # Archimedes spiral: "r = a + b*theta", where "a" is the minimum radius of the
        # outer spiral and "2*pi*b" is the spacing between the lines pairs.
        theta_max: float = (r - self.a_spiral) / self.b_spiral
        theta_min: float = max(theta_max - (n_turns * 2 * np.pi), 0)
        assert theta_max > 0, "thetas max/min must be positive!"
        outer_spiral_r_max: float = r
        outer_spiral_r_min: float = self.a_spiral + self.b_spiral * theta_min
        inner_spiral_r_max: float = outer_spiral_r_max - self.line_width / 2
        inner_spiral_r_min: float = outer_spiral_r_min - self.line_width / 2

        # Calculate the total spiral length (sum of outer and inner spirals)
        # by numerical integration w/r to the radius.
        L: float = (
            integrate.quad(
                func=self._line_element,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
            )[0]
            + integrate.quad(
                func=self._line_element,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
            )[0]
        )

        # Calculate propagation losses in the spiral
        gamma: float = self.models.gamma_of_u(h)
        alpha_prop: float = self.models.alpha_wg + (gamma * self.models.alpha_fluid)
        prop_losses_spiral: float = alpha_prop * L

        # Calculate bending losses in the spiral by integration w/r to radius
        # (sum for inner and outer spiral losses)
        bend_losses_spiral: float = (
            integrate.quad(
                func=self._line_element_bend_loss,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
                args=(h,),
            )[0]
            + integrate.quad(
                func=self._line_element_bend_loss,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
                args=(h,),
            )[0]
        )

        # Approximate the "joint" between the two parallel waveguides in the spiral
        # at the center by:
        # - Outer waveguide: Half circle of radius equal to the outer
        #                    spiral minimum radius (length = L_HC)
        # - Inner waveguide: "S-bend" with 2 half circles of radii equal to half
        #                    the minimum radii of the inner spiral (length = L_SB1)
        #                    and the outer spiral (length = L_SB2).
        L_HC: float = np.pi * outer_spiral_r_min
        L_SB1: float = np.pi * (inner_spiral_r_min / 2)
        L_SB2: float = np.pi * (outer_spiral_r_min / 2)
        prop_losses_joint: float = alpha_prop * (L_HC + L_SB1 + L_SB2)
        bend_losses_joint: float = (
            self.models.alpha_bend(r=outer_spiral_r_min, u=h) * L_HC
            + self.models.alpha_bend(r=inner_spiral_r_min / 2, u=h) * L_SB1
            + self.models.alpha_bend(r=outer_spiral_r_min / 2, u=h) * L_SB2
        )

        # Total losses in the spiral
        total_losses: float = (
            prop_losses_spiral
            + bend_losses_spiral
            + prop_losses_joint
            + bend_losses_joint
        )
        a2: float = np.e**-total_losses

        # Calculate the sensitivity of the spiral
        L_total: float = L + L_HC + L_SB1 + L_SB2
        Snr: float = (4 * np.pi / self.models.lambda_res) * L_total * gamma * a2

        return Snr, outer_spiral_r_min, L_total

    def _obj_fun(self, x, r: float) -> float:
        """
        Objective function for the optimization in find_max_sensitivity()
        """

        # Fetch the solution vector components
        h: float = x[0]
        n_turns: float = x[1]

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        h = min(h, self.models.u_domain_max)
        h = max(h, self.models.u_domain_min)
        n_turns = max(n_turns, 0)

        # Calculate sensitivity at current solution vector S(r, h, n_turns)
        s = self._calc_sensitivity(r=r, h=h, n_turns=n_turns)[0]
        assert s >= 0, "S should not be negative!"

        return -s / 1000

    def _find_max_sensitivity(
        self, r: float
    ) -> tuple[float, float, float, float, float, float]:
        """
        Calculate maximum sensitivity at r over all h and n_turns
        """

        # Determine search domain extrema for h
        h_min, h_max = self.models.u_search_domain(r)

        # Determine search domain extrema for the numer of turns in the spiral
        n_turns_max: float = (r - self.a_spiral) / self.line_width
        n_turns_max = min(n_turns_max, self.turns_max)
        n_turns_max = max(n_turns_max, self.turns_min)

        # Only proceed with the minimization if the radius at which the solution is
        # sought is greater than the minimum allowed outer spiral radius
        if r >= self.a_spiral + self.line_width:
            # If this is the first optimization, set the initial guesses for h at the
            # maximum value in the domain and the numbers of turns at the minimum
            # value (at small radii, bending losses are high, the optimal solution
            # will be at high h and low number of turns),else use previous solution.
            if np.any(self.previous_solution == -1):
                h0: float = h_max
                n_turns_0: float = self.turns_min
            else:
                h0, n_turns_0 = self.previous_solution

            # Find h and n_turns that maximize S at radius R
            optimization_result = optimize.minimize(
                fun=self._obj_fun,
                x0=np.asarray([h0, n_turns_0]),
                bounds=((h_min, h_max), (self.turns_min, n_turns_max)),
                args=(r,),
                method="Powell",
                options={"ftol": 1e-9},
            )
            h_max_S = optimization_result["x"][0]
            n_turns_max_S = optimization_result["x"][1]

            # Calculate maximum sensitivity at the solution
            S, outer_spiral_r_min, L = self._calc_sensitivity(
                r=r, h=h_max_S, n_turns=n_turns_max_S
            )

            # Update previous solution
            self.previous_solution = np.asarray([h_max_S, n_turns_max_S])

        else:
            h_max_S = h_max
            n_turns_max_S = 0
            S = 1
            outer_spiral_r_min = 0
            L = 0

            # Update previous solution
            self.previous_solution = np.asarray([h_max, self.turns_min])

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(h_max_S) * 100

        return S, h_max_S, n_turns_max_S, outer_spiral_r_min, L, gamma

    def analyze(self):
        """
        Analyse the sensor performance for all radii in the R domain

        :return: None
        """
        # Analyse the sensor performance for all radii in the R domain
        self.results = [self._find_max_sensitivity(r=r) for r in self.models.R]

        # Unpack the analysis results as a function of radius into separate lists, the
        # order must be the same as in the find_max_sensitivity() return statement above
        [
            self.S,
            self.h,
            self.n_turns,
            self.outer_spiral_r_min,
            self.L,
            self.gamma,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_S = np.amax(self.S)
        self.max_S_radius = self.models.R[np.argmax(self.S)]

        # If dynamic range of mode solver data exceeded for the spiral, show warning
        inner_spiral_r_min: float = (
            min(self.outer_spiral_r_min[self.outer_spiral_r_min > 0]) - self.line_width
        )
        if inner_spiral_r_min < self.models.R_data_min:
            self.logger(
                f"{Fore.YELLOW}WARNING: Minimum spiral bend radius "
                + f"({inner_spiral_r_min:.2f} um) is below minimum value in mode solver"
                + f" data ({self.models.R_data_min:.2f} um)!{Style.RESET_ALL}"
            )

        # Console message
        self.logger("Spiral sensor analysis done.")
