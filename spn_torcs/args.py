def init_parser(parser):
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--recording', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video-folder', type=str, default='videos')
    parser.add_argument('--branch-multiplier', type=int, default=3)
    parser.add_argument('--min-branch-width', type=int, default=10)
    parser.add_argument('--consistency-factor', type=float, default=20)

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--bar-min', type=int, default=200)
    parser.add_argument('--expert-ratio', type=float, default=0.05)
    parser.add_argument('--safe-length-collision', type=int, default=30)
    parser.add_argument('--safe-length-offroad', type=int, default=15)
    parser.add_argument('--bin-divide', type=list, default=[5, 5])

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--frame-history-len', type=int, default=3)
    parser.add_argument('--pred-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--save-path', type=str, default='mpc_15_step')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--num-total-act', type=int, default=2)
    parser.add_argument('--epsilon-frames', type=int, default=50000)
    parser.add_argument('--learning-starts', type=int, default=100)
    parser.add_argument('--learning-freq', type=int, default=100)
    parser.add_argument('--target-update-freq', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num-train-steps', type=int, default=10)
    # enviroument configurations
    parser.add_argument('--env', type=str, default='torcs-v0', metavar='ENV', help='environment')
    parser.add_argument('--simple-seg', action='store_true')
    parser.add_argument('--xvfb', type=bool, default=True)
    parser.add_argument('--game-config', type=str, default='/media/shared/pyTORCS/game_config/michigan.xml')
    parser.add_argument('--continuous', action='store_true')

    # model configurations
    parser.add_argument('--sample-based-planning', type=bool, default=True)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--drn-model', type=str, default='dla46x_c')
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--one-hot', action='store_true')

    parser.add_argument('--use-collision', action='store_true')
    parser.add_argument('--use-offroad', action='store_true')
    parser.add_argument('--use-otherlane', action='store_true')
    parser.add_argument('--use-distance', action='store_true')
    parser.add_argument('--use-pos', action='store_true')
    parser.add_argument('--use-angle', action='store_true')
    parser.add_argument('--use-speed', action='store_true')
    parser.add_argument('--use-seg', action='store_true')
    parser.add_argument('--use-xyz', action='store_true')
    parser.add_argument('--use-dqn', action='store_true')
    parser.add_argument('--use-random-reset', action='store_true')
    parser.add_argument('--num-dqn-action', type=int, default=10)

    parser.add_argument('--sample-with-pos', action='store_true')
    parser.add_argument('--sample-with-angle', action='store_true')
    parser.add_argument('--sample-with-offroad', action='store_true')
    parser.add_argument('--sample-with-otherlane', action='store_true')
    parser.add_argument('--sample-with-collision', action='store_true')
    parser.add_argument('--sample-with-speed', action='store_true')
    parser.add_argument('--sample-with-distance', action='store_true')
    parser.add_argument('--num-same-step', type=int, default=1)

    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--lstm2', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--info-dim', type=int, default=32)

    parser.add_argument('--target-pos', type=float, default=0)
    parser.add_argument('--target-speed', type=float, default=-1)
    parser.add_argument('--target-dist', type=float, default=-1)
