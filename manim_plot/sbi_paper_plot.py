from manim import *
from classes import *
import numpy as np
from scipy.stats import norm, uniform


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# setup
background_color = "#FFFFFF"
font = "Helvetica"
tex_font = TexFontTemplates.helvetica_fourier_it


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Functions

def get_priors():
    priors = np.array([[-2.30, 0.30], [-2.89, 0.30], [-0.30, 0.30], [0.55, 0.10], [0.50, 0.10] , [1.0, 13.8]])
    labels_in = [r'$\alpha_{IMF}$', r'$\log_{10}(N_0)$', r'$\log_{10}(SFE)$', r'$\log_{10}(SFR_{peak})$' , r'$x_{out}$', r'$T$']
    #labels_in = ["a", "b", "c", "d", "e", "f"]
    labels_out = ['C', 'Fe', 'H', 'He', 'Mg', 'N', 'Ne', 'O', 'Si']

    return priors , labels_in, labels_out


def load_abundances():
    data = np.load('../../ChempyMulti/tutorial_data/TNG_Training_Data.npz', mmap_mode='r')
    abun = np.load("../data/abun.npy")
    time = np.load("../data/time.npy")
    #el = data['elements']

    return abun, time#, el

def load_posterior():
    alpha_IMF_obs = np.load("../data/alpha_IMF_obs.npy")
    log10_N_Ia_obs = np.load("../data/log10_N_Ia_obs.npy")

    return alpha_IMF_obs, log10_N_Ia_obs

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 1
class Plot1(Scene):
    def construct(self):
        self.camera.background_color = background_color
        #ScreenRectangle(aspect_ratio=10, height=4)

        # -------------------
        # Step 1: Priors & Simulator

        def prior_box():
            priors, labels_in, labels_out = get_priors()

            prior_text = Text("Priors", font=font, color=BLACK).scale(0.4)

            plots = []
            for i in range(len(priors)):
                # make a 2d normal distribution for each parameter
                mu, sigma = priors[i]


                if i == 5:
                    x = np.linspace(mu-1, sigma+1, 1000)
                    y = uniform.pdf(x, loc=mu, scale=sigma-mu)
                    axes = Axes(
                        x_range=[mu-1, sigma+1],
                        y_range=[0, y.max()],
                        axis_config={"color": BLACK, "include_ticks":False},
                        tips=True
                    ).set_stroke(width=1)

                    graph = axes.plot_line_graph(x, y, line_color=ORANGE, add_vertex_dots=False).set_stroke(width=1)

                else:
                    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
                    y = norm.pdf(x, mu, sigma)
                    axes = Axes(
                        x_range=[mu - 5*sigma, mu + 5*sigma],
                        y_range=[0, y.max()],
                        axis_config={"color": BLACK, "include_ticks":False},
                        tips=True
                    ).set_stroke(width=1)

                    graph = axes.plot(lambda x: norm.pdf(x, mu, sigma), color=ORANGE).set_stroke(width=1)

                #x_label = axes.get_x_axis_label(Tex(labels_in[i]), edge=DOWN, direction=DOWN, buff=10).set_color(BLACK)
                x_label = Tex(labels_in[i]).scale(3).set_color(BLACK).move_to(axes.x_axis.get_center() + 1.5*DOWN)

                plot = VGroup(axes, graph, x_label).scale(0.06)
                plot = VGroup(plot, x_label).move_to([(i%3)*1, -(int(i/3))*0.6, 0])
                plots.append(plot)

            plots = VGroup(*plots)
            prior_text.next_to(plots, UP, buff=0.1).align_to(plots, LEFT)
            priors = VGroup(plots, prior_text)

            w = priors.width
            h = priors.height

            box = SurroundingRectangle(priors, color="#2F2827", corner_radius=0.15, buff=0.2)

            priors = VGroup(box, priors)

            return priors, w, h

        def Chempy_box():
            chempy_text = Tex(r"$CHEMPY$", color=BLACK).scale(1.1)
            sim_text = Text(r"Simulator", font=font ,color=BLACK).scale(0.4).next_to(chempy_text, UP, buff=0.2).align_to(chempy_text, LEFT)
            chempy_text = VGroup(chempy_text, sim_text)
            box = SurroundingRectangle(chempy_text, color="#2F2827", corner_radius=0.15, buff=0.2)
            chempy = VGroup(box, chempy_text)

            return chempy

        priors, w, h = prior_box()
        self.add(priors.to_edge(LEFT, buff=0.1))

        chempy = Chempy_box()
        self.add(chempy.scale_to_fit_width(priors.width).next_to(priors, UP, buff=0.2).align_to(priors, LEFT))

        step_one = VGroup(priors, chempy).move_to([-5,0,0]).to_edge(LEFT, buff=0.1)
        one = Tex(r"\}", color="#2F2827").stretch_to_fit_height(step_one.height).next_to(step_one, RIGHT, buff=0.2)
        self.add(one)


        # -------------------
        # Step 2: NN Simulator

        def nn_box(w, h):
            nn_text = Text(r"NN Simulator", font=font, color=BLACK).scale(0.4)

            NN = NeuralNetworkMobject([2, 5, 5, 2], output_neuron_color="#475389", input_neuron_color=ORANGE).scale_to_fit_width(w)
            text = Text("NN Simulator", font=font, color=BLACK).scale(0.4).next_to(NN, UP, buff=0.2).align_to(NN, LEFT)

            NN = VGroup(NN, text).scale_to_fit_height(h)
            box = SurroundingRectangle(NN, color="#2F2827", corner_radius=0.15, buff=0.2)
            NN = VGroup(box, NN)

            w = NN.width
            h = NN.height

            return NN, w, h

        nn,w,h = nn_box(w,h)
        self.add(nn.next_to(one, RIGHT, buff=0.2))



        two = Arrow([0,0,0], [1.5,0,0], color="#2F2827").next_to(nn, RIGHT, buff=0.2)
        self.add(two)

        # -------------------
        # Step 3: Simulate Data

        def sim_box(w, h):
            abun, time = load_abundances()
            abun = np.delete(abun[:,0], 2, axis=1).T

            sim_text = Text(r"Simulated Data", font=font, color=BLACK).scale(0.4)

            axs = Axes(
                x_range=[0, time.max()+1],
                y_range=[abun.min()-0.5, abun.max()+0.5],
                axis_config={"color": BLACK, "include_ticks":False},
                tips=True
            ).set_stroke(width=1)

            lines = []
            colors = ["#7285db", "#6474c0", "#5664a4", "#475389", "#39436e", "#2b3252", "#1c2137", "#0e111b"]
            for i in range(abun.shape[0]):
                graph = axs.plot_line_graph(time, abun[i], line_color=colors[i], add_vertex_dots=False).set_stroke(width=1)
                lines.append(graph)

            graph = VGroup(*lines)
            x_label = Tex(r"$Time$").scale(3).set_color(BLACK).move_to(axs.x_axis.get_center() + 3 * DOWN)
            y_label = Tex(r"$Abundances$").scale(2).set_color(BLACK).rotate(np.pi/2).move_to(axs.y_axis.get_center() + 1 * LEFT)

            plot = VGroup(axs, graph, x_label, y_label).scale_to_fit_width(w)
            sim_text.next_to(plot, UP, buff=0.1).align_to(plot, LEFT)
            plot = VGroup(plot, sim_text).scale_to_fit_width(w)

            w, h = plot.width, plot.height

            box = SurroundingRectangle(plot, color="#2F2827", corner_radius=0.15, buff=0.2)
            #box.surround(plot, buff=1)
            plot = VGroup(box, plot)

            return plot, w, h

        sim, w, h = sim_box(w,h)
        self.add(sim.next_to(two, RIGHT, buff=0.2))

        three = Arrow([0,0,0], [1.5,0,0], color="#2F2827").next_to(sim, RIGHT, buff=0.2)
        self.add(three)

        # -------------------
        # Step 4: SBI Model

        def sbi_box(w, h):
            sbi_text = Text(r"SBI Model", font=font, color=BLACK).scale(0.4)

            NN = NeuralNetworkMobject([2, 5, 5, 2]).scale_to_fit_width(w)
            text = Text("Neural Density Estimator", font=font, color=BLACK).scale(0.4).next_to(NN, UP, buff=0.2)#.align_to(NN, LEFT)

            NN = VGroup(NN, text).scale_to_fit_height(h)
            box = SurroundingRectangle(NN, color="#2F2827", corner_radius=0.15, buff=0.2)
            NN = VGroup(box, NN)

            return NN

        sbi = sbi_box(w,h)
        self.add(sbi.next_to(three, RIGHT, buff=0.2))

        four = Arrow([0,0,0], [1.5,0,0], color="#2F2827").next_to(sbi, RIGHT, buff=0.2)
        self.add(four)

        # -------------------
        # Step 5: Posterior

        def post_box(w,h):

            alpha_IMF_obs, log10_N_Ia_obs = load_posterior()
            priors, labels_in, labels_out = get_priors()

            axs = Axes(
                x_range=[-1,1],
                y_range=[-1,1],
                axis_config={"color": BLACK, "include_ticks":False},
                tips=True
            ).set_stroke(width=1)

            points = VGroup(* [Dot(color=ORANGE, fill_opacity=0.3, radius=0.05).move_to([10*(alpha_IMF_obs[i]-priors[0][0]), 10*(log10_N_Ia_obs[i]-priors[1][0]), 0]) for i in range(10000)])

            x_label = Tex(r"$\alpha_{IMF}$").scale(2).set_color(BLACK)
            x_label.move_to(axs.x_axis.get_corner(RIGHT) + 1*DOWN + LEFT*x_label.get_width()/2)
            y_label = Tex(r"$\log_{10}(N_{Ia})$").scale(2).set_color(BLACK)
            y_label.move_to(axs.y_axis.get_corner(UP) + LEFT*y_label.get_width()/2+0.5*LEFT)

            plot = VGroup(axs, points, x_label, y_label).scale_to_fit_height(h)
            text = Text("Posterior", font=font, color=BLACK).scale(0.4).next_to(plot, UP, buff=0.1).align_to(plot, LEFT)
            plot = VGroup(plot, text).scale_to_fit_height(h)

            box = SurroundingRectangle(plot, color="#2F2827", corner_radius=0.15, buff=0.2)
            plot = VGroup(box, plot)

            return plot

        post = post_box(w,h)
        self.add(post.next_to(four, RIGHT, buff=0.2))

        # -------------------
        # group all steps
        loop = VGroup(sim, two, sbi)
        #box = SurroundingRectangle(loop, color="#2F2827", corner_radius=0.15, buff=0.15)
        text = Tex(r"$*N_{stars}$", color=BLACK).scale_to_fit_width(three.width) #.scale(0.8).next_to(loop, UP, buff=0.2).align_to(loop, RIGHT)
        text.next_to(four, UP, buff=0.1).move_to(four.get_center() + 0.4*UP)
        #text.align_to(three, RIGHT)
        #text.next_to(sbi, RIGHT, buff=0.2)
        #text.move_to(box.get_corner(UR) + 0.2*UP + LEFT*text.width)
        #self.add(box)
        self.add(text)
        #post.scale_to_fit_height(box.height).next_to(three, RIGHT, buff=0.2)

        steps = VGroup(step_one, one, nn, two, sim, three, sbi, four, text, post)
        # adjust to screen width
        steps.scale_to_fit_width(14)
        steps.move_to(ORIGIN)





