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
        models: Models,
        parameters: dict,
        logger: Callable = print,
    ):

        # Load class instance parameter values
        self.spacing: float = parameters["spiral_spacing"]
        self.turns_min: float = parameters["spiral_turns_min"]
        self.turns_max: float = parameters["spiral_turns_max"]
        self.models: Models = models
        self.logger: Callable = logger

        # Calculate spiral parameters
        self.line_width_min: float = 2 * (
            (
                self.models.core_v_value
                if self.models.core_v_name == "w"
                else self.models.u_domain_min
            )
            + self.spacing
        )
        self.a_spiral_min: float = 1.5 * self.line_width_min
        self.b_spiral_min: float = self.line_width_min / (2 * np.pi)

        # Declare class instance internal variables
        self.previous_solution: np.ndarray = np.asarray([-1, -1])

        # Declare class instance result variables and arrays
        self.S: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.n_turns: np.ndarray = np.ndarray([])
        self.outer_spiral_r_min: np.ndarray = np.ndarray([])
        self.L: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.a2: np.ndarray = np.ndarray([])
        self.max_S: float = 0
        self.max_S_radius: float = 0
        self.results: list = []

    def _calc_spiral_parameters(self, w: float) -> tuple[float, float, float]:
        """
        Calculate spiral parameters @ r, w, and n_turns
        """

        line_width: float = 2 * (w + self.spacing)
        b: float = line_width / (2 * np.pi)
        a: float = 1.5 * line_width

        return line_width, a, b

    @staticmethod
    def _plot_arc(
        thetas: np.ndarray,
        r: np.ndarray | float,
        x0: float = 0,
        y0: float = 0,
        theta0: float = 0,
        spiral_x: np.ndarray = np.asarray([]),
        spiral_y: np.ndarray = np.asarray([]),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate parametric arc theta(n) & r(n) centered at [x0,y0] and rotated by
        theta0, append arc x[n]/y[n] arc sample coordinates to x/y spiral arrays
        """

        # Build array of radii by duplication if r is a float
        rs: np.ndarray = r * np.ones(len(thetas)) if isinstance(r, float) else r

        # Rotate the arc by "theta0" so that consecutive spirals are aligned
        x_rot: np.ndarray = x0 + rs * np.cos(thetas)
        y_rot: np.ndarray = y0 + rs * np.sin(thetas)
        x: np.ndarray = x_rot * np.cos(theta0) - y_rot * np.sin(theta0)
        y: np.ndarray = y_rot * np.cos(theta0) + x_rot * np.sin(theta0)

        # return length and updated x,y arrays
        return np.append(spiral_x, x), np.append(spiral_y, y)

    def draw_spiral(
        self,
        r_outer: float,
        h: float,
        w: float,
        n_turns: float,
        r_window: float,
        figure: plt.Figure = None,
    ) -> tuple[plt.Figure, dict,]:
        """
        This function actually draws a spiral!
        """

        # Archimedes spiral parameters
        line_width, a_spiral, b_spiral = self._calc_spiral_parameters(w=w)

        # Define new figure if required, else use axes passed as a function parameter
        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.axes[0]
            ax.clear()

        # Outer spiral
        theta_max: float = (r_outer - (a_spiral + self.spacing + w)) / b_spiral
        theta_min: float = max(theta_max - (n_turns * 2 * np.pi), 0)
        thetas_spiral: np.ndarray = np.linspace(theta_max, theta_min, 1000)
        r_outer_spiral_inner: np.ndarray = (a_spiral + self.spacing) + (
            b_spiral * thetas_spiral
        )
        theta0: float = -theta_max + np.pi / 2
        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_spiral,
            r=r_outer_spiral_inner,
            theta0=theta0,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_spiral,
            r=r_outer_spiral_inner + w,
            theta0=theta0,
        )

        # Inner spiral
        r_inner_spiral_inner: np.ndarray = a_spiral + (b_spiral * thetas_spiral)
        inner_spiral_x_in, inner_spiral_y_in = self._plot_arc(
            thetas=thetas_spiral,
            r=r_inner_spiral_inner,
            theta0=theta0,
        )
        inner_spiral_x_out, inner_spiral_y_out = self._plot_arc(
            thetas=thetas_spiral,
            r=r_inner_spiral_inner + w,
            theta0=theta0,
        )

        # Joint: half circle between outer waveguide and S-bend
        thetas_half_circle: np.ndarray = np.linspace(theta_min, theta_min - np.pi, 100)
        r_joint_inner: np.ndarray = (a_spiral + self.spacing) + (
            b_spiral * thetas_half_circle
        )
        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_half_circle,
            r=r_joint_inner,
            theta0=theta0,
            spiral_x=outer_spiral_x_in,
            spiral_y=outer_spiral_y_in,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_half_circle,
            r=r_joint_inner + w,
            theta0=theta0,
            spiral_x=outer_spiral_x_out,
            spiral_y=outer_spiral_y_out,
        )

        # Joint: half S-bend between outer waveguide half-circle and origin
        thetas_S_bend_outer: np.ndarray = np.linspace(
            thetas_half_circle[-1], thetas_half_circle[-1] - np.pi, 100
        )
        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_S_bend_outer,
            r=(r_joint_inner[-1] - w / 2) / 2,
            x0=-((r_joint_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=-((r_joint_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=outer_spiral_x_in,
            spiral_y=outer_spiral_y_in,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_S_bend_outer,
            r=(r_joint_inner[-1] + 3 * w / 2) / 2,
            x0=-((r_joint_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=-((r_joint_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=outer_spiral_x_out,
            spiral_y=outer_spiral_y_out,
        )

        # Joint: half S-bend between inner waveguide and origin
        thetas_S_bend_inner = np.linspace(
            thetas_spiral[-1], thetas_spiral[-1] - np.pi, 100
        )
        inner_spiral_x_in, inner_spiral_y_in = self._plot_arc(
            thetas=thetas_S_bend_inner,
            r=(r_inner_spiral_inner[-1] - w / 2) / 2,
            x0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=inner_spiral_x_in,
            spiral_y=inner_spiral_y_in,
        )
        inner_spiral_x_out, inner_spiral_y_out = self._plot_arc(
            thetas=thetas_S_bend_inner,
            r=(r_inner_spiral_inner[-1] + 3 * w / 2) / 2,
            x0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=inner_spiral_x_out,
            spiral_y=inner_spiral_y_out,
        )

        # Total spiral length (inner + outer spirals) calculated by finite differences
        # (sum of averaged outer/inner edges for both spirals)
        outer_spiral_length: float = (
            np.sum(
                np.sqrt(
                    np.diff(outer_spiral_x_out) ** 2 + np.diff(outer_spiral_y_out) ** 2
                )
            )
            + np.sum(
                np.sqrt(
                    np.diff(outer_spiral_x_in) ** 2 + np.diff(outer_spiral_y_in) ** 2
                )
            )
        ) / 2
        inner_spiral_length: float = (
            np.sum(
                np.sqrt(
                    np.diff(inner_spiral_x_out) ** 2 + np.diff(inner_spiral_y_out) ** 2
                )
            )
            + np.sum(
                np.sqrt(
                    np.diff(inner_spiral_x_in) ** 2 + np.diff(inner_spiral_y_in) ** 2
                )
            )
        ) / 2
        spiral_length: float = outer_spiral_length + inner_spiral_length

        # Build dictionary of spiral waveguide vertex coordinates
        spiral_waveguide_coordinates = {
            "outer_spiral_x_out": outer_spiral_x_out,
            "outer_spiral_y_out": outer_spiral_y_out,
            "outer_spiral_x_in": outer_spiral_x_in,
            "outer_spiral_y_in": outer_spiral_y_in,
            "inner_spiral_x_out": inner_spiral_x_out,
            "inner_spiral_y_out": inner_spiral_y_out,
            "inner_spiral_x_in": inner_spiral_x_in,
            "inner_spiral_y_in": inner_spiral_y_in,
        }

        # Plot info & formatting
        S, _, L, _ = self._calc_sensitivity(
            r=r_outer, u=h if self.models.core_v_name == "w" else w, n_turns=n_turns
        )
        ax.set_aspect("equal")
        ax.set_xlim(-r_window, r_window)
        ax.set_ylim(-r_window, r_window)
        ax.set_title(
            "Archimedes spiral : "
            + f"{n_turns: .2f} turns, "
            + f"w = {self.models.core_v_value:.3f} μm, "
            + f"spacing = {self.spacing:.1f} μm, "
            + f"h = {h:.3f} μm, "
            + rf"S = {S:.0f} RIU$^{{-1}}$"
            + f"\nR = {r_outer:.1f} μm, "
            + rf"R$_{{min}}$ = {r_joint_inner[-1]:.1f} μm, "
            + f"S-bend radius = {(r_joint_inner[-1] - w / 2) / 2:.1f} μm, "
            + f"L = {L:.1f} ({spiral_length:.2f} by finite diffs) μm"
        )
        ax.plot(outer_spiral_x_in, outer_spiral_y_in, color="red")
        ax.plot(outer_spiral_x_out, outer_spiral_y_out, color="red")
        ax.plot(inner_spiral_x_in, inner_spiral_y_in, color="blue")
        ax.plot(inner_spiral_x_out, inner_spiral_y_out, color="blue")
        ax.set_xlabel(r"$\mu$m")
        ax.set_ylabel(r"$\mu$m")

        return (
            fig,
            spiral_waveguide_coordinates,
        )

    def _line_element(self, r: float, w: float) -> float:
        """
        Line element for numerical integration in polar coordinates (r, theta)
        "dl = sqrt(r**2 + (dr/dtheta)**2)*dtheta", converted to a function of "r" only
        with the spiral equation (r = a + b*theta) to "dl(r) = sqrt((r/b)**2 + 1)*dr".
        """

        _, _, b_spiral = self._calc_spiral_parameters(w=w)

        return np.sqrt((r / b_spiral) ** 2 + 1)

    def _line_element_bend_loss(self, r: float, u: float, w: float) -> float:
        """
        Bending losses for a line element: alpha_bend(r)*dl(r)
        """

        return self.models.alpha_bend(r=r, u=u) * self._line_element(r=r, w=w)

    def _calc_sensitivity(
        self, r: float, u: float, n_turns: float
    ) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core height & height
        and number of turns
        """

        # Determine waveguide core width & height
        if self.models.core_v_name == "w":
            h = u
            w = self.models.core_v_value
        else:
            h = self.models.core_v_value
            w = u

        # Archimedes spiral: "r = a + b*theta", where "a" is the minimum radius of the
        # outer spiral and "2*pi*b" is the spacing between the lines pairs.
        # Check that r is above the allowed minimum.
        line_width, a_spiral, b_spiral = self._calc_spiral_parameters(w=w)
        a_spiral_outer: float = a_spiral + self.spacing + w

        # Spiral radii and angle extrema
        theta_max: float = (r - a_spiral_outer) / b_spiral
        theta_min: float = theta_max - (n_turns * 2 * np.pi)
        if theta_min < 0:
            return 1.0e-10, 0, 0, 0

        # Spiral radii extrema
        outer_spiral_r_max: float = r
        outer_spiral_r_min: float = a_spiral_outer + b_spiral * theta_min
        inner_spiral_r_max: float = outer_spiral_r_max - line_width / 2
        inner_spiral_r_min: float = outer_spiral_r_min - line_width / 2

        # Calculate the total spiral length (sum of outer and inner spirals)
        # by numerical integration w/r to the radius.
        L: float = (
            integrate.quad(
                func=self._line_element,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
                args=(w,),
            )[0]
            + integrate.quad(
                func=self._line_element,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
                args=(w,),
            )[0]
        )

        # Calculate propagation losses in the spiral
        gamma: float = self.models.gamma_of_u(
            h if self.models.core_v_name == "w" else w
        )
        alpha_prop: float = self.models.alpha_wg(u=u) + (
            gamma * self.models.alpha_fluid
        )
        prop_losses_spiral: float = alpha_prop * L

        # Calculate bending losses in the spiral by integration w/r to radius
        # (sum for inner and outer spiral losses)
        bend_losses_spiral: float = (
            integrate.quad(
                func=self._line_element_bend_loss,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
                args=(
                    u,
                    w,
                ),
            )[0]
            + integrate.quad(
                func=self._line_element_bend_loss,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
                args=(
                    u,
                    w,
                ),
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
            self.models.alpha_bend(r=outer_spiral_r_min, u=u) * L_HC
            + self.models.alpha_bend(r=inner_spiral_r_min / 2, u=u) * L_SB1
            + self.models.alpha_bend(r=outer_spiral_r_min / 2, u=u) * L_SB2
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

        return Snr, outer_spiral_r_min, L_total, a2

    def _obj_fun(self, x, r: float) -> float:
        """
        Objective function for the optimization in find_max_sensitivity()
        """

        # Fetch the solution vector components
        u: float = x[0]
        n_turns: float = x[1]

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.u_domain_max)
        u = max(u, self.models.u_domain_min)
        n_turns = max(n_turns, 0)

        # Calculate sensitivity at current solution vector S(r, u, n_turns)
        s, _, _, _ = self._calc_sensitivity(r=r, u=u, n_turns=n_turns)
        assert s >= 0, "S should not be negative!"

        return -s / 1000

    def _find_max_sensitivity(
        self, r: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Calculate maximum sensitivity at r over all h and n_turns
        """

        # Determine search domain extrema for u
        u_min, u_max = self.models.u_search_domain(r)

        # Spiral properties at minimum core width
        w_min: float = (
            self.models.core_v_value if self.models.core_v_name == "w" else u_min
        )
        line_width_min, a_spiral_min, b_spiral_min = self._calc_spiral_parameters(
            w=w_min
        )
        theta_min: float = 2 * np.pi * self.turns_min
        r_min: float = (a_spiral_min + self.spacing + w_min) + theta_min * b_spiral_min

        # Determine search domain maximum for the numer of turns in the spiral
        n_turns_max: float = (r - r_min) / line_width_min
        n_turns_max = max(n_turns_max, self.turns_min)
        n_turns_max = min(n_turns_max, self.turns_max)

        # Only proceed with the minimization if the radius at which the solution is
        # sought is greater than the minimum allowed outer spiral radius.
        if r >= r_min + line_width_min * self.turns_min:
            # If this is the first optimization, set the initial guesses for u at the
            # maximum value in the domain and the numbers of turns at the minimum
            # value (at small radii, bending losses are high, the optimal solution
            # will be at high u and low number of turns),else use previous solution.
            if np.any(self.previous_solution == -1):
                u0: float = u_max
                n_turns_0: float = self.turns_min
            else:
                u0, n_turns_0 = self.previous_solution

            # Find h and n_turns that maximize S at radius R
            optimization_result = optimize.minimize(
                fun=self._obj_fun,
                x0=np.asarray([u0, n_turns_0]),
                bounds=((u_min, u_max), (self.turns_min, n_turns_max)),
                args=(r,),
                method="Powell",
                options={"ftol": 1e-9},
            )
            u_max_S = optimization_result["x"][0]
            n_turns_max_S = optimization_result["x"][1]

            # Calculate maximum sensitivity at the solution
            S, outer_spiral_r_min, L, a2 = self._calc_sensitivity(
                r=r, u=u_max_S, n_turns=n_turns_max_S
            )

            # Update previous solution
            self.previous_solution = np.asarray([u_max_S, n_turns_max_S])

        else:
            u_max_S = u_max
            n_turns_max_S = 0
            S = 1
            outer_spiral_r_min = 0
            L = 0
            a2 = 0

            # Update previous solution
            self.previous_solution = np.asarray([u_max, self.turns_min])

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(u_max_S) * 100

        return S, u_max_S, n_turns_max_S, outer_spiral_r_min, L, gamma, a2

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
            self.u,
            self.n_turns,
            self.outer_spiral_r_min,
            self.L,
            self.gamma,
            self.a2,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_S = np.amax(self.S)
        self.max_S_radius = self.models.R[np.argmax(self.S)]

        # If dynamic range of mode solver data exceeded for the spiral, show warning
        inner_spiral_r_min: float = (
            min(self.outer_spiral_r_min[self.outer_spiral_r_min > 0])
            - self.line_width_min
        )
        if inner_spiral_r_min < self.models.R_data_min:
            self.logger(
                f"{Fore.YELLOW}WARNING: Minimum spiral bend radius "
                + f"({inner_spiral_r_min:.2f} um) is below minimum value in mode solver"
                + f" data ({self.models.R_data_min:.2f} um)!{Style.RESET_ALL}"
            )

        # Console message
        self.logger("Spiral sensor analysis done.")
